/**
 * @file chirp.c
 * @brief Chirp DSL - A Scheme-like language with Smalltalk-style keywords for DSP pipelines
 * 
 * Chirp allows describing complex DSP pipelines using S-expressions:
 *   (pipeline 
 *     (fft :size 1024) 
 *     twiddle 
 *     (bfly 4)
 *     (lift :predict gaussian :update softmax)
 *     (custom softmax)
 *     reduce-sum)
 * 
 * @version 1.0.0
 */

#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

/* Builtin registry */
#define CHIRP_MAX_BUILTINS 512
#define CHIRP_MAX_INST 1024
#define CHIRP_MAX_SYMBOL 128

typedef struct {
    char *name;
    void (*fn)(void);
} chirp_builtin;

chirp_builtin chirp_table[CHIRP_MAX_BUILTINS];
int g_chirp_count = 0;
static int chirp_initialized = 0;

/* Token types for the lexer */
typedef enum {
    CHIRP_TOK_LPAREN,      /* ( */
    CHIRP_TOK_RPAREN,      /* ) */
    CHIRP_TOK_KEYWORD,     /* :size, :predict, etc. */
    CHIRP_TOK_SYMBOL,      /* fft, twiddle, gaussian, etc. */
    CHIRP_TOK_NUMBER,      /* 1024, 4, etc. */
    CHIRP_TOK_STRING,      /* "hello" */
    CHIRP_TOK_EOF,         /* end of input */
    CHIRP_TOK_ERROR        /* lexical error */
} chirp_token_type;

/* Token structure */
typedef struct {
    chirp_token_type type;
    char *text;            /* Raw text */
    int value;             /* For numbers */
    int line;
    int col;
} chirp_token;

/* Lexer state */
typedef struct {
    const char *src;
    size_t pos;
    size_t len;
    int line;
    int col;
} chirp_lexer;

/* AST Node types */
typedef enum {
    CHIRP_NODE_PIPELINE,   /* (pipeline ...) */
    CHIRP_NODE_FFT,        /* (fft :size N) */
    CHIRP_NODE_TWIDDLE,    /* twiddle */
    CHIRP_NODE_BFLY,       /* (bfly N) */
    CHIRP_NODE_LIFT,       /* (lift :predict X :update Y) */
    CHIRP_NODE_CUSTOM,     /* (custom name) */
    CHIRP_NODE_REDUCE,     /* reduce-sum, reduce-max, etc. */
    CHIRP_NODE_LITERAL,    /* number or symbol */
    CHIRP_NODE_LIST        /* generic list */
} chirp_node_type;

/* AST Node */
typedef struct chirp_node {
    chirp_node_type type;
    char *sym;             /* Symbol name */
    int value;             /* Numeric value */
    struct chirp_node **children;
    int n_children;
    int cap_children;
    /* Keyword arguments (Smalltalk-style) */
    char **kw_names;
    struct chirp_node **kw_values;
    int n_kw;
} chirp_node;

/* Forward declarations */
static chirp_token chirp_lexer_next(chirp_lexer *lex);
static chirp_node* chirp_parse_expr(chirp_lexer *lex);
static void chirp_node_free(chirp_node *node);
static faf_inst chirp_compile_node(chirp_node *node, faf_transform *t, int *inst_count);

/* Initialize builtin registry */
static void chirp_init_builtins(void) {
    if (chirp_initialized) return;
    memset(chirp_table, 0, sizeof(chirp_table));
    g_chirp_count = 0;
    chirp_initialized = 1;
}

/* Public API: Free all registered builtin names and reset */
void chirp_cleanup(void) {
    for (int i = 0; i < g_chirp_count; i++) {
        free(chirp_table[i].name);
        chirp_table[i].name = NULL;
        chirp_table[i].fn = NULL;
    }
    g_chirp_count = 0;
    chirp_initialized = 0;
}

/* Public API: Return the number of currently registered builtins */
int chirp_count(void) {
    return g_chirp_count;
}

/* Public API: Register a custom function */
int chirp_register(const char *name, void (*fn)(void)) {
    chirp_init_builtins();
    if (g_chirp_count >= CHIRP_MAX_BUILTINS) return -1;
    if (!name || !fn) return -1;
    
    /* Check if already registered */
    for (int i = 0; i < g_chirp_count; i++) {
        if (strcmp(chirp_table[i].name, name) == 0) {
            chirp_table[i].fn = fn;
            return i;
        }
    }
    
    chirp_table[g_chirp_count].name = strdup(name);
    chirp_table[g_chirp_count].fn = fn;
    return g_chirp_count++;
}

/* Look up a builtin by name */
static int chirp_lookup_builtin(const char *name) {
    for (int i = 0; i < g_chirp_count; i++) {
        if (strcmp(chirp_table[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/* Create a lexer */
static chirp_lexer chirp_lexer_new(const char *src) {
    chirp_lexer lex = {
        .src = src,
        .pos = 0,
        .len = strlen(src),
        .line = 1,
        .col = 1
    };
    return lex;
}

/* Get current char */
static char chirp_lexer_peek(chirp_lexer *lex) {
    if (lex->pos >= lex->len) return '\0';
    return lex->src[lex->pos];
}

/* Advance and return char */
static char chirp_lexer_advance(chirp_lexer *lex) {
    if (lex->pos >= lex->len) return '\0';
    char c = lex->src[lex->pos];
    lex->pos++;
    if (c == '\n') {
        lex->line++;
        lex->col = 1;
    } else {
        lex->col++;
    }
    return c;
}

/* Skip whitespace and comments */
static void chirp_lexer_skip_ws(chirp_lexer *lex) {
    while (1) {
        char c = chirp_lexer_peek(lex);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            chirp_lexer_advance(lex);
        } else if (c == ';') {
            /* Line comment */
            while (chirp_lexer_peek(lex) != '\n' && chirp_lexer_peek(lex) != '\0') {
                chirp_lexer_advance(lex);
            }
        } else {
            break;
        }
    }
}

/* Read a symbol/keyword */
static char* chirp_lexer_read_symbol(chirp_lexer *lex, int *is_keyword) {
    size_t start = lex->pos;
    *is_keyword = (chirp_lexer_peek(lex) == ':');
    
    if (*is_keyword) {
        chirp_lexer_advance(lex); /* skip : */
    }
    
    while (isalnum(chirp_lexer_peek(lex)) || 
           chirp_lexer_peek(lex) == '-' || 
           chirp_lexer_peek(lex) == '_' ||
           chirp_lexer_peek(lex) == '?') {
        chirp_lexer_advance(lex);
    }
    
    size_t len = lex->pos - start - (*is_keyword ? 1 : 0);
    char *sym = malloc(len + 1);
    strncpy(sym, lex->src + start + (*is_keyword ? 1 : 0), len);
    sym[len] = '\0';
    return sym;
}

/* Read a number */
static int chirp_lexer_read_number(chirp_lexer *lex) {
    int val = 0;
    while (isdigit(chirp_lexer_peek(lex))) {
        val = val * 10 + (chirp_lexer_advance(lex) - '0');
    }
    return val;
}

/* Get next token */
static chirp_token chirp_lexer_next(chirp_lexer *lex) {
    chirp_lexer_skip_ws(lex);
    
    chirp_token tok = {0};
    tok.line = lex->line;
    tok.col = lex->col;
    
    char c = chirp_lexer_peek(lex);
    
    if (c == '\0') {
        tok.type = CHIRP_TOK_EOF;
        tok.text = strdup("<eof>");
    } else if (c == '(') {
        tok.type = CHIRP_TOK_LPAREN;
        tok.text = strdup("(");
        chirp_lexer_advance(lex);
    } else if (c == ')') {
        tok.type = CHIRP_TOK_RPAREN;
        tok.text = strdup(")");
        chirp_lexer_advance(lex);
    } else if (c == ':') {
        /* Keyword like :size, :predict */
        tok.type = CHIRP_TOK_KEYWORD;
        int dummy;
        tok.text = chirp_lexer_read_symbol(lex, &dummy);
    } else if (isdigit(c)) {
        tok.type = CHIRP_TOK_NUMBER;
        tok.value = chirp_lexer_read_number(lex);
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", tok.value);
        tok.text = strdup(buf);
    } else if (isalpha(c) || c == '-' || c == '_' || c == '*') {
        /* Symbol */
        int is_kw = 0;
        tok.type = CHIRP_TOK_SYMBOL;
        tok.text = chirp_lexer_read_symbol(lex, &is_kw);
    } else {
        tok.type = CHIRP_TOK_ERROR;
        char buf[2] = {c, '\0'};
        tok.text = strdup(buf);
        chirp_lexer_advance(lex);
    }
    
    return tok;
}

/* Free a token */
static void chirp_token_free(chirp_token *tok) {
    free(tok->text);
    tok->text = NULL;
}

/* Create a new AST node */
static chirp_node* chirp_node_new(chirp_node_type type) {
    chirp_node *node = calloc(1, sizeof(chirp_node));
    node->type = type;
    node->cap_children = 4;
    node->children = calloc(node->cap_children, sizeof(chirp_node*));
    return node;
}

/* Add child to node */
static void chirp_node_add_child(chirp_node *parent, chirp_node *child) {
    if (parent->n_children >= parent->cap_children) {
        parent->cap_children *= 2;
        parent->children = realloc(parent->children, 
                                   parent->cap_children * sizeof(chirp_node*));
    }
    parent->children[parent->n_children++] = child;
}

/* Add keyword argument (Smalltalk-style) */
static void chirp_node_add_kwarg(chirp_node *node, const char *name, chirp_node *value) {
    if (node->n_kw == 0) {
        node->kw_names = calloc(8, sizeof(char*));
        node->kw_values = calloc(8, sizeof(chirp_node*));
    }
    /* Simple fixed array for now */
    if (node->n_kw < 8) {
        node->kw_names[node->n_kw] = strdup(name);
        node->kw_values[node->n_kw] = value;
        node->n_kw++;
    }
}

/* Get keyword argument value */
static chirp_node* chirp_node_get_kwarg(chirp_node *node, const char *name) {
    for (int i = 0; i < node->n_kw; i++) {
        if (strcmp(node->kw_names[i], name) == 0) {
            return node->kw_values[i];
        }
    }
    return NULL;
}

/* Free an AST node */
static void chirp_node_free(chirp_node *node) {
    if (!node) return;
    free(node->sym);
    for (int i = 0; i < node->n_children; i++) {
        chirp_node_free(node->children[i]);
    }
    free(node->children);
    for (int i = 0; i < node->n_kw; i++) {
        free(node->kw_names[i]);
        chirp_node_free(node->kw_values[i]);
    }
    free(node->kw_names);
    free(node->kw_values);
    free(node);
}

/* Parse an expression */
static chirp_node* chirp_parse_expr(chirp_lexer *lex) {
    chirp_token tok = chirp_lexer_next(lex);
    chirp_node *node = NULL;
    
    switch (tok.type) {
        case CHIRP_TOK_LPAREN: {
            /* List expression: (operator args...) */
            chirp_token op = chirp_lexer_next(lex);
            if (op.type != CHIRP_TOK_SYMBOL && op.type != CHIRP_TOK_KEYWORD) {
                fprintf(stderr, "Chirp: expected symbol after '('\n");
                chirp_token_free(&op);
                chirp_token_free(&tok);
                return NULL;
            }
            
            /* Determine node type from operator */
            if (strcmp(op.text, "pipeline") == 0) {
                node = chirp_node_new(CHIRP_NODE_PIPELINE);
            } else if (strcmp(op.text, "fft") == 0) {
                node = chirp_node_new(CHIRP_NODE_FFT);
            } else if (strcmp(op.text, "bfly") == 0) {
                node = chirp_node_new(CHIRP_NODE_BFLY);
            } else if (strcmp(op.text, "lift") == 0) {
                node = chirp_node_new(CHIRP_NODE_LIFT);
            } else if (strcmp(op.text, "custom") == 0) {
                node = chirp_node_new(CHIRP_NODE_CUSTOM);
            } else {
                node = chirp_node_new(CHIRP_NODE_LIST);
                node->sym = strdup(op.text);
            }
            
            chirp_token_free(&op);
            
            /* Parse arguments */
            while (1) {
                chirp_lexer_skip_ws(lex);
                char c = chirp_lexer_peek(lex);
                if (c == ')') {
                    chirp_lexer_advance(lex);
                    break;
                }
                if (c == '\0') {
                    fprintf(stderr, "Chirp: unexpected EOF\n");
                    break;
                }
                
                /* Check for keyword argument */
                if (c == ':') {
                    int dummy;
                    char *kw_name = chirp_lexer_read_symbol(lex, &dummy);
                    chirp_node *kw_value = chirp_parse_expr(lex);
                    if (kw_value) {
                        chirp_node_add_kwarg(node, kw_name, kw_value);
                    }
                    free(kw_name);
                } else {
                    chirp_node *child = chirp_parse_expr(lex);
                    if (child) {
                        chirp_node_add_child(node, child);
                    }
                }
            }
            break;
        }
        
        case CHIRP_TOK_SYMBOL: {
            /* Bare symbol - check for registered builtins first */
            if (strcmp(tok.text, "twiddle") == 0) {
                node = chirp_node_new(CHIRP_NODE_TWIDDLE);
            } else if (strcmp(tok.text, "reduce-sum") == 0) {
                node = chirp_node_new(CHIRP_NODE_REDUCE);
                node->sym = strdup("sum");
            } else if (strcmp(tok.text, "reduce-max") == 0) {
                node = chirp_node_new(CHIRP_NODE_REDUCE);
                node->sym = strdup("max");
            } else if (strcmp(tok.text, "reduce-min") == 0) {
                node = chirp_node_new(CHIRP_NODE_REDUCE);
                node->sym = strdup("min");
            } else if (chirp_lookup_builtin(tok.text) >= 0) {
                /* Registered builtin function - first class! */
                node = chirp_node_new(CHIRP_NODE_CUSTOM);
                node->sym = strdup(tok.text);
            } else {
                node = chirp_node_new(CHIRP_NODE_LITERAL);
                node->sym = strdup(tok.text);
            }
            break;
        }
        
        case CHIRP_TOK_NUMBER: {
            node = chirp_node_new(CHIRP_NODE_LITERAL);
            node->value = tok.value;
            break;
        }
        
        default:
            fprintf(stderr, "Chirp: unexpected token '%s' at line %d\n", 
                    tok.text, tok.line);
            break;
    }
    
    chirp_token_free(&tok);
    return node;
}

/* Print AST for debugging */
static void chirp_node_print(chirp_node *node, int indent) {
    if (!node) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    
    switch (node->type) {
        case CHIRP_NODE_PIPELINE: printf("(pipeline"); break;
        case CHIRP_NODE_FFT: printf("(fft"); break;
        case CHIRP_NODE_TWIDDLE: printf("twiddle"); break;
        case CHIRP_NODE_BFLY: printf("(bfly"); break;
        case CHIRP_NODE_LIFT: printf("(lift"); break;
        case CHIRP_NODE_CUSTOM: printf("(custom"); break;
        case CHIRP_NODE_REDUCE: printf("(reduce-%s", node->sym ? node->sym : "?"); break;
        case CHIRP_NODE_LITERAL: 
            if (node->sym) printf("'%s'", node->sym);
            else printf("%d", node->value);
            break;
        case CHIRP_NODE_LIST: printf("(%s", node->sym ? node->sym : "list"); break;
    }
    
    /* Print keyword args */
    for (int i = 0; i < node->n_kw; i++) {
        printf(" :%s ", node->kw_names[i]);
        chirp_node_print(node->kw_values[i], 0);
    }
    
    /* Print children */
    if (node->n_children > 0) {
        printf("\n");
        for (int i = 0; i < node->n_children; i++) {
            chirp_node_print(node->children[i], indent + 1);
        }
        for (int i = 0; i < indent; i++) printf("  ");
    }
    
    if (node->n_children > 0 || node->type == CHIRP_NODE_PIPELINE ||
        node->type == CHIRP_NODE_FFT || node->type == CHIRP_NODE_BFLY ||
        node->type == CHIRP_NODE_LIFT || node->type == CHIRP_NODE_CUSTOM ||
        node->type == CHIRP_NODE_LIST) {
        printf(")");
    }
    printf("\n");
}

/* Helper to emit an instruction */
static void chirp_emit_inst(faf_transform *t, faf_inst inst, int *inst_count) {
    if (t->code && *inst_count < CHIRP_MAX_INST) {
        t->code[*inst_count] = inst;
    }
    (*inst_count)++;
}

/* Compile an AST node to IR instructions */
static void chirp_compile_node_emit(chirp_node *node, faf_transform *t, int *inst_count);

static void chirp_compile_node_emit(chirp_node *node, faf_transform *t, int *inst_count) {
    if (!node) return;
    
    faf_inst inst = {0};
    
    switch (node->type) {
        case CHIRP_NODE_PIPELINE: {
            /* Pipeline just sequences its children */
            for (int i = 0; i < node->n_children; i++) {
                chirp_compile_node_emit(node->children[i], t, inst_count);
            }
            return;
        }
        
        case CHIRP_NODE_FFT: {
            inst.packed = FAF_FFT_STAGE;
            /* Get :size keyword */
            chirp_node *size_node = chirp_node_get_kwarg(node, "size");
            if (size_node && size_node->type == CHIRP_NODE_LITERAL) {
                inst.a0 = size_node->value;
            } else if (node->n_children > 0 && node->children[0]->type == CHIRP_NODE_LITERAL) {
                inst.a0 = node->children[0]->value;
            } else {
                inst.a0 = 64; /* default */
            }
            t->type = FAF_TRANSFORM_FFT;
            if (inst.a0 > t->n) t->n = inst.a0;
            break;
        }
        
        case CHIRP_NODE_TWIDDLE: {
            inst.packed = FAF_TWIDDLE_MUL;
            inst.a0 = 0;
            inst.a1 = 0;
            break;
        }
        
        case CHIRP_NODE_BFLY: {
            int radix = 2;
            if (node->n_children > 0 && node->children[0]->type == CHIRP_NODE_LITERAL) {
                radix = node->children[0]->value;
            }
            switch (radix) {
                case 2: inst.packed = FAF_BFLY2; break;
                case 4: inst.packed = FAF_BFLY4; break;
                case 8: inst.packed = FAF_BFLY8; break;
                default: inst.packed = FAF_BFLY2; break;
            }
            inst.a0 = 0;
            break;
        }
        
        case CHIRP_NODE_LIFT: {
            /* Lifting scheme with :predict and :update keywords */
            chirp_node *predict_node = chirp_node_get_kwarg(node, "predict");
            chirp_node *update_node = chirp_node_get_kwarg(node, "update");
            
            if (predict_node && predict_node->sym) {
                faf_inst pred_inst = {0};
                pred_inst.packed = FAF_LIFT_PRED;
                pred_inst.a0 = chirp_lookup_builtin(predict_node->sym);
                if (pred_inst.a0 < 0) pred_inst.a0 = 0;
                chirp_emit_inst(t, pred_inst, inst_count);
            }
            
            if (update_node && update_node->sym) {
                faf_inst upd_inst = {0};
                upd_inst.packed = FAF_LIFT_UPD;
                upd_inst.a0 = chirp_lookup_builtin(update_node->sym);
                if (upd_inst.a0 < 0) upd_inst.a0 = 0;
                chirp_emit_inst(t, upd_inst, inst_count);
            }
            return;
        }
        
        case CHIRP_NODE_CUSTOM: {
            /* Support both: (custom name) and bare function name */
            const char *fn_name = NULL;
            if (node->sym) {
                fn_name = node->sym;  /* First-class: just 'name' */
            } else if (node->n_children > 0 && node->children[0]->sym) {
                fn_name = node->children[0]->sym;  /* Old syntax: (custom name) */
            }
            
            if (fn_name) {
                inst.packed = FAF_CALL_BUILTIN;
                inst.a0 = chirp_lookup_builtin(fn_name);
                if (inst.a0 < 0) {
                    fprintf(stderr, "Chirp: unknown builtin '%s'\n", fn_name);
                    inst.a0 = 0;
                }
            }
            break;
        }
        
        case CHIRP_NODE_REDUCE: {
            if (strcmp(node->sym, "sum") == 0) {
                inst.packed = FAF_REDUCE_SUM;
            } else if (strcmp(node->sym, "max") == 0) {
                inst.packed = FAF_REDUCE_MAX;
            } else if (strcmp(node->sym, "min") == 0) {
                inst.packed = FAF_REDUCE_MIN;
            }
            inst.a0 = 0;
            break;
        }
        
        default:
            return;
    }
    
    chirp_emit_inst(t, inst, inst_count);
}

/* Public API: Compile a Chirp program */
faf_transform* chirp_compile(const char *source) {
    if (!source) return NULL;
    
    chirp_init_builtins();
    
    /* Parse the source */
    chirp_lexer lex = chirp_lexer_new(source);
    chirp_node *ast = chirp_parse_expr(&lex);
    
    if (!ast) {
        fprintf(stderr, "Chirp: failed to parse program\n");
        return NULL;
    }
    
    /* Create transform */
    faf_transform *t = calloc(1, sizeof(faf_transform));
    t->precision = FAF_PREC_FP32;
    t->n = 64; /* default size */
    
    /* First pass: count instructions */
    int inst_count = 0;
    chirp_compile_node_emit(ast, t, &inst_count);
    
    /* Allocate code array (+1 for END) */
    t->n_inst = inst_count + 1;
    t->code = calloc(t->n_inst, sizeof(faf_inst));
    
    /* Second pass: generate instructions */
    inst_count = 0;
    chirp_compile_node_emit(ast, t, &inst_count);
    
    /* Add END opcode */
    faf_inst end_inst = {0};
    end_inst.packed = FAF_END;
    t->code[inst_count] = end_inst;
    
    /* Cleanup AST */
    chirp_node_free(ast);
    
    return t;
}
