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
 * @version 1.1.0
 */

#include "chirp.h"
#include "faf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

/* Builtin registry */
#define CHIRP_MAX_BUILTINS 512
#define CHIRP_MAX_INST 65536
#define CHIRP_MAX_SYMBOL 128

typedef struct {
    char *name;
    void (*fn)(void);
    void (*fn_f64)(void);
    int kind;
    void *ctx;
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
    int value;             /* For integers */
    double fvalue;         /* For floats */
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
    CHIRP_NODE_RFFT,       /* (rfft :size N) */
    CHIRP_NODE_IRFFT,      /* (irfft :size N) */
    CHIRP_NODE_SPECTRAL,   /* (spectral name) */
    CHIRP_NODE_BANDPASS,   /* (bandpass :lo :hi) */
    CHIRP_NODE_MULSPEC,    /* (mul-spectrum) */
    CHIRP_NODE_CONJ,       /* conj */
    CHIRP_NODE_TWIDDLE,    /* twiddle */
    CHIRP_NODE_BFLY,       /* (bfly N) */
    CHIRP_NODE_LIFT,       /* (lift :predict X :update Y) */
    CHIRP_NODE_CUSTOM,     /* (custom name) */
    CHIRP_NODE_REDUCE,     /* reduce-sum, reduce-max, etc. */
    CHIRP_NODE_DWT,        /* (dwt :family haar :size N :levels L) */
    CHIRP_NODE_IDWT,       /* (idwt ...) */
    CHIRP_NODE_THRESHOLD,  /* (threshold :mode soft :lambda x) */
    CHIRP_NODE_CWT,        /* (cwt :n N :wavelet morse ...) */
    CHIRP_NODE_ICWT,       /* (icwt :n N ...) */
    CHIRP_NODE_LITERAL,    /* number or symbol */
    CHIRP_NODE_LIST        /* generic list */
} chirp_node_type;

/* AST Node */
typedef struct chirp_node {
    chirp_node_type type;
    char *sym;             /* Symbol name */
    int value;             /* Integer value */
    double fvalue;         /* Floating value */
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
        chirp_table[i].fn_f64 = NULL;
        chirp_table[i].kind = CHIRP_KIND_UNARY;
        chirp_table[i].ctx = NULL;
    }
    g_chirp_count = 0;
    chirp_initialized = 0;
}

/* Public API: Return the number of currently registered builtins */
int chirp_count(void) {
    return g_chirp_count;
}

/* Public API: Return the name of a registered builtin by index */
const char* chirp_builtin_name(int idx) {
    if (idx < 0 || idx >= g_chirp_count) return NULL;
    return chirp_table[idx].name;
}

/* Public API: Return the function pointer of a registered builtin by index */
void* chirp_builtin_fn(int idx) {
    if (idx < 0 || idx >= g_chirp_count) return NULL;
    return (void*)chirp_table[idx].fn;
}

/* Public API: Return the number of registered builtins (alias for chirp_count) */
int chirp_builtin_count(void) {
    return g_chirp_count;
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
    chirp_table[g_chirp_count].fn_f64 = NULL;
    chirp_table[g_chirp_count].kind = CHIRP_KIND_UNARY;
    chirp_table[g_chirp_count].ctx = NULL;
    return g_chirp_count++;
}

int chirp_register_ex(const char *name, void (*fn)(void), int kind) {
    int id = chirp_register(name, fn);
    if (id >= 0) chirp_table[id].kind = kind;
    return id;
}

int chirp_register_unary(const char *name, float (*f32)(float),
                         double (*f64)(double)) {
    if (!name || !f32) return -1;
    int id = chirp_register(name, (void (*)(void))f32);
    if (id < 0) return id;
    chirp_table[id].kind = CHIRP_KIND_UNARY;
    chirp_table[id].fn_f64 = (void (*)(void))f64;
    return id;
}

int chirp_register_spectral(const char *name, chirp_spectral_fn fn, void *ctx) {
    return chirp_register_spectral_ex(name, fn, NULL, ctx);
}

int chirp_register_spectral_ex(const char *name, chirp_spectral_fn f32,
                               chirp_spectral_fn_f64 f64, void *ctx) {
    if (!name || !f32) return -1;
    int id = chirp_register(name, (void (*)(void))f32);
    if (id < 0) return id;
    chirp_table[id].kind = CHIRP_KIND_SPECTRAL;
    chirp_table[id].fn_f64 = (void (*)(void))f64;
    chirp_table[id].ctx = ctx;
    return id;
}

void *chirp_builtin_ctx(int idx) {
    if (idx < 0 || idx >= g_chirp_count) return NULL;
    return chirp_table[idx].ctx;
}

int chirp_bind(faf_transform *t, const char *name, void *re, void *im,
               size_t n_bins) {
    if (!t || !name || !re) {
        faf_set_error("chirp_bind: transform, name, and re are required");
        return -1;
    }
    if (strcmp(name, "H") != 0 && strcmp(name, "mul-spectrum") != 0 &&
        strcmp(name, "kernel") != 0) {
        faf_set_error("chirp_bind: unknown slot '%s' (use \"H\")", name);
        return -1;
    }
    t->user_aux = re;
    t->user_aux_im = im;
    t->user_aux_n = n_bins;
    return 0;
}

int chirp_builtin_kind(int idx) {
    if (idx < 0 || idx >= g_chirp_count) return CHIRP_KIND_OTHER;
    return chirp_table[idx].kind;
}

/* Public API: Return the function pointer for a builtin, selecting the
 * precision-appropriate variant (e.g. sin_f32 vs sin_f64) when available. */
void* chirp_builtin_fn_for_precision(int idx, faf_precision precision) {
    if (idx < 0 || idx >= g_chirp_count) return NULL;
    if (precision == FAF_PREC_FP64 && chirp_table[idx].fn_f64)
        return (void *)chirp_table[idx].fn_f64;

    const char *name = chirp_builtin_name(idx);
    if (!name) return NULL;

    /* If the registered name already carries a precision suffix, use it. */
    if (strstr(name, "_f32") || strstr(name, "_f64")) {
        return chirp_builtin_fn(idx);
    }

    /* Bare alias: try to find the variant matching the transform precision. */
    const char *suffix = (precision == FAF_PREC_FP64) ? "_f64" : "_f32";
    char suffixed[256];
    int n = snprintf(suffixed, sizeof(suffixed), "%s%s", name, suffix);
    if (n > 0 && (size_t)n < sizeof(suffixed)) {
        int new_idx = chirp_lookup_builtin(suffixed);
        if (new_idx >= 0) return chirp_builtin_fn(new_idx);
    }

    /* Fallback: return the original registered function. */
    return chirp_builtin_fn(idx);
}

void chirp_apply_split_f32(const faf_transform *t, uint32_t a0, uint32_t a1,
                           uint32_t a2, float *re, float *im, size_t n) {
    if (!re || n == 0) return;
    if (a0 == CHIRP_OP_CONJ) {
        if (!im) return;
        for (size_t i = 0; i < n; i++) im[i] = -im[i];
        return;
    }
    if (a0 == CHIRP_OP_MUL) {
        if (!im || !t || !t->user_aux || !t->user_aux_im) return;
        size_t m = t->user_aux_n < n ? t->user_aux_n : n;
        const float *hr = (const float *)t->user_aux;
        const float *hi = (const float *)t->user_aux_im;
        for (size_t i = 0; i < m; i++) {
            float ar = re[i], ai = im[i];
            re[i] = ar * hr[i] - ai * hi[i];
            im[i] = ar * hi[i] + ai * hr[i];
        }
        return;
    }
    if (a0 == CHIRP_OP_BANDPASS) {
        if (!im) return;
        uint32_t lo = a1, hi = a2;
        for (size_t i = 0; i < n; i++) {
            if (i < lo || i > hi) {
                re[i] = 0.0f;
                im[i] = 0.0f;
            }
        }
        return;
    }
    int idx = (int)a0;
    int kind = chirp_builtin_kind(idx);
    void *fn = chirp_builtin_fn_for_precision(idx, FAF_PREC_FP32);
    void *ctx = chirp_builtin_ctx(idx);
    if (!fn) return;
    if (kind == CHIRP_KIND_UNARY) {
        for (size_t i = 0; i < n; i++)
            re[i] = ((float (*)(float))fn)(re[i]);
    } else if (kind == CHIRP_KIND_COMPLEX) {
        if (!im) return;
        chirp_complex_fn cf = (chirp_complex_fn)fn;
        for (size_t i = 0; i < n; i++)
            cf(&re[i], &im[i]);
    } else if (kind == CHIRP_KIND_VECTOR) {
        ((chirp_vector_fn)fn)(re, n);
    } else if (kind == CHIRP_KIND_SPECTRAL || kind == CHIRP_KIND_HERMITIAN) {
        if (!im) return;
        ((chirp_spectral_fn)fn)(re, im, n, ctx);
        if (kind == CHIRP_KIND_HERMITIAN) {
            im[0] = 0.0f;
            if (n > 0) im[n - 1] = 0.0f;
        }
    }
}

void chirp_apply_split_f64(const faf_transform *t, uint32_t a0, uint32_t a1,
                           uint32_t a2, double *re, double *im, size_t n) {
    if (!re || n == 0) return;
    if (a0 == CHIRP_OP_CONJ) {
        if (!im) return;
        for (size_t i = 0; i < n; i++) im[i] = -im[i];
        return;
    }
    if (a0 == CHIRP_OP_MUL) {
        if (!im || !t || !t->user_aux || !t->user_aux_im) return;
        size_t m = t->user_aux_n < n ? t->user_aux_n : n;
        const double *hr = (const double *)t->user_aux;
        const double *hi = (const double *)t->user_aux_im;
        for (size_t i = 0; i < m; i++) {
            double ar = re[i], ai = im[i];
            re[i] = ar * hr[i] - ai * hi[i];
            im[i] = ar * hi[i] + ai * hr[i];
        }
        return;
    }
    if (a0 == CHIRP_OP_BANDPASS) {
        if (!im) return;
        uint32_t lo = a1, hi = a2;
        for (size_t i = 0; i < n; i++) {
            if (i < lo || i > hi) {
                re[i] = 0.0;
                im[i] = 0.0;
            }
        }
        return;
    }
    int idx = (int)a0;
    int kind = chirp_builtin_kind(idx);
    void *fn = chirp_builtin_fn_for_precision(idx, FAF_PREC_FP64);
    void *ctx = chirp_builtin_ctx(idx);
    if (!fn) return;
    if (kind == CHIRP_KIND_UNARY) {
        for (size_t i = 0; i < n; i++)
            re[i] = ((double (*)(double))fn)(re[i]);
    } else if (kind == CHIRP_KIND_COMPLEX) {
        if (!im) return;
        chirp_complex_fn_f64 cf = (chirp_complex_fn_f64)fn;
        for (size_t i = 0; i < n; i++)
            cf(&re[i], &im[i]);
    } else if (kind == CHIRP_KIND_VECTOR) {
        ((chirp_vector_fn_f64)fn)(re, n);
    } else if (kind == CHIRP_KIND_SPECTRAL || kind == CHIRP_KIND_HERMITIAN) {
        if (!im) return;
        chirp_spectral_fn_f64 sf = (chirp_spectral_fn_f64)fn;
        /* f64 pointer may be missing; fall back is the f32 slot, skip */
        if (chirp_table[idx].fn_f64)
            sf(re, im, n, ctx);
        if (kind == CHIRP_KIND_HERMITIAN) {
            im[0] = 0.0;
            if (n > 0) im[n - 1] = 0.0;
        }
    }
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

/* Read a number (integer or floating). Sets *out_int and *out_dbl. */
static void chirp_lexer_read_number(chirp_lexer *lex, int *out_int, double *out_dbl) {
    int sign = 1;
    if (chirp_lexer_peek(lex) == '-') {
        sign = -1;
        chirp_lexer_advance(lex);
    }
    double val = 0.0;
    while (isdigit(chirp_lexer_peek(lex))) {
        val = val * 10.0 + (double)(chirp_lexer_advance(lex) - '0');
    }
    if (chirp_lexer_peek(lex) == '.') {
        chirp_lexer_advance(lex);
        double place = 0.1;
        while (isdigit(chirp_lexer_peek(lex))) {
            val += place * (double)(chirp_lexer_advance(lex) - '0');
            place *= 0.1;
        }
    }
    val *= (double)sign;
    *out_dbl = val;
    *out_int = (int)val;
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
    } else if (isdigit(c) || (c == '-' && lex->pos + 1 < lex->len &&
                              (isdigit(lex->src[lex->pos + 1]) ||
                               lex->src[lex->pos + 1] == '.'))) {
        tok.type = CHIRP_TOK_NUMBER;
        chirp_lexer_read_number(lex, &tok.value, &tok.fvalue);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.8g", tok.fvalue);
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
            } else if (strcmp(op.text, "fft") == 0 || strcmp(op.text, "ifft") == 0) {
                node = chirp_node_new(CHIRP_NODE_FFT);
                if (strcmp(op.text, "ifft") == 0)
                    node->value = 1; /* inverse */
            } else if (strcmp(op.text, "rfft") == 0) {
                node = chirp_node_new(CHIRP_NODE_RFFT);
            } else if (strcmp(op.text, "irfft") == 0) {
                node = chirp_node_new(CHIRP_NODE_IRFFT);
            } else if (strcmp(op.text, "spectral") == 0) {
                node = chirp_node_new(CHIRP_NODE_SPECTRAL);
            } else if (strcmp(op.text, "bandpass") == 0) {
                node = chirp_node_new(CHIRP_NODE_BANDPASS);
            } else if (strcmp(op.text, "mul-spectrum") == 0) {
                node = chirp_node_new(CHIRP_NODE_MULSPEC);
            } else if (strcmp(op.text, "conj") == 0) {
                node = chirp_node_new(CHIRP_NODE_CONJ);
            } else if (strcmp(op.text, "bfly") == 0) {
                node = chirp_node_new(CHIRP_NODE_BFLY);
            } else if (strcmp(op.text, "lift") == 0) {
                node = chirp_node_new(CHIRP_NODE_LIFT);
            } else if (strcmp(op.text, "custom") == 0) {
                node = chirp_node_new(CHIRP_NODE_CUSTOM);
            } else if (strcmp(op.text, "dwt") == 0) {
                node = chirp_node_new(CHIRP_NODE_DWT);
            } else if (strcmp(op.text, "idwt") == 0) {
                node = chirp_node_new(CHIRP_NODE_IDWT);
            } else if (strcmp(op.text, "threshold") == 0) {
                node = chirp_node_new(CHIRP_NODE_THRESHOLD);
            } else if (strcmp(op.text, "cwt") == 0) {
                node = chirp_node_new(CHIRP_NODE_CWT);
            } else if (strcmp(op.text, "icwt") == 0) {
                node = chirp_node_new(CHIRP_NODE_ICWT);
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
                    faf_set_error("Chirp: unexpected EOF");
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
            } else if (strcmp(tok.text, "conj") == 0) {
                node = chirp_node_new(CHIRP_NODE_CONJ);
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
            node->fvalue = tok.fvalue;
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
        case CHIRP_NODE_RFFT: printf("(rfft"); break;
        case CHIRP_NODE_IRFFT: printf("(irfft"); break;
        case CHIRP_NODE_SPECTRAL: printf("(spectral"); break;
        case CHIRP_NODE_BANDPASS: printf("(bandpass"); break;
        case CHIRP_NODE_MULSPEC: printf("(mul-spectrum"); break;
        case CHIRP_NODE_CONJ: printf("conj"); break;
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

static const char* chirp_node_name(chirp_node *node) {
    if (!node) return NULL;
    if (node->sym) return node->sym;
    if (node->n_children > 0 && node->children[0] && node->children[0]->sym)
        return node->children[0]->sym;
    return NULL;
}

static int chirp_node_int(chirp_node *node, int fallback) {
    if (!node) return fallback;
    if (node->type == CHIRP_NODE_LITERAL) return node->value;
    return fallback;
}

static double chirp_node_float(chirp_node *node, double fallback) {
    if (!node) return fallback;
    if (node->type == CHIRP_NODE_LITERAL) {
        return (node->fvalue != 0.0 || node->value == 0) ? node->fvalue : (double)node->value;
    }
    return fallback;
}

static int chirp_parse_family(chirp_node *node, faf_wavelet_family *out) {
    const char *name = chirp_node_name(node);
    if (!name) return -1;
    return faf_wavelet_from_name(name, out);
}

static int chirp_parse_layout(chirp_node *node, faf_layout *out) {
    const char *name = chirp_node_name(node);
    if (!name) return -1;
    if (strcmp(name, "hermitian") == 0)      { *out = FAF_LAYOUT_HERMITIAN; return 0; }
    if (strcmp(name, "interleaved") == 0)    { *out = FAF_LAYOUT_INTERLEAVED; return 0; }
    if (strcmp(name, "split") == 0)          { *out = FAF_LAYOUT_SPLIT; return 0; }
    if (strcmp(name, "real") == 0)           { *out = FAF_LAYOUT_REAL; return 0; }
    if (strcmp(name, "default") == 0)        { *out = FAF_LAYOUT_DEFAULT; return 0; }
    return -1;
}

static int chirp_parse_norm(chirp_node *node, faf_norm *out) {
    const char *name = chirp_node_name(node);
    if (!name) return -1;
    if (strcmp(name, "none") == 0)           { *out = FAF_NORM_NONE; return 0; }
    if (strcmp(name, "ortho") == 0)          { *out = FAF_NORM_ORTHO; return 0; }
    if (strcmp(name, "forward") == 0)        { *out = FAF_NORM_FORWARD; return 0; }
    if (strcmp(name, "lazy") == 0)           { *out = FAF_NORM_LAZY; return 0; }
    if (strcmp(name, "jpeg2000") == 0)       { *out = FAF_NORM_JPEG2000; return 0; }
    if (strcmp(name, "default") == 0)        { *out = FAF_NORM_DEFAULT; return 0; }
    return -1;
}

static size_t chirp_log2_size(size_t n) {
    size_t bits = 0;
    while (n > 1) { n >>= 1; bits++; }
    return bits;
}

static void chirp_ensure_fft_twiddles(faf_transform *t, size_t n, int inverse) {
    if (t->twiddles[0] || n == 0) return;
    int full = !faf_is_power_of_2(n);
    if (t->precision == FAF_PREC_FP64) {
        t->twiddles[0] = malloc((full ? n * 2 : n) * sizeof(double));
        if (t->twiddles[0]) {
            if (full) {
                faf_gen_twiddles_full_f64((double *)t->twiddles[0], n, inverse);
                t->twiddle_sizes[0] = n;
            } else {
                faf_gen_twiddles_f64((double *)t->twiddles[0], n, inverse);
                t->twiddle_sizes[0] = n / 2;
            }
        }
    } else {
        t->twiddles[0] = malloc((full ? n * 2 : n) * sizeof(float));
        if (t->twiddles[0]) {
            if (full) {
                faf_gen_twiddles_full_f32((float *)t->twiddles[0], n, inverse);
                t->twiddle_sizes[0] = n;
            } else {
                faf_gen_twiddles_f32((float *)t->twiddles[0], n, inverse);
                t->twiddle_sizes[0] = n / 2;
            }
        }
    }
}

static void chirp_emit_dwt_stages(faf_transform *t, int *inst_count,
                                  faf_wavelet_family family, size_t n,
                                  size_t levels, int inverse) {
    uint32_t flags = inverse ? FAF_DWT_FLAG_INVERSE : 0;
    if (inverse) {
        for (size_t lv = levels; lv-- > 0; ) {
            size_t length = n >> lv;
            if (length < 2) continue;
            faf_inst st = {0};
            st.packed = FAF_DWT_STAGE;
            st.a0 = (uint32_t)family;
            st.a1 = (uint32_t)length;
            st.a2 = flags;
            chirp_emit_inst(t, st, inst_count);
        }
    } else {
        for (size_t lv = 0; lv < levels; lv++) {
            size_t length = n >> lv;
            if (length < 2) break;
            faf_inst st = {0};
            st.packed = FAF_DWT_STAGE;
            st.a0 = (uint32_t)family;
            st.a1 = (uint32_t)length;
            st.a2 = flags;
            chirp_emit_inst(t, st, inst_count);
        }
    }
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
        
        case CHIRP_NODE_RFFT:
        case CHIRP_NODE_IRFFT:
        case CHIRP_NODE_SPECTRAL:
        case CHIRP_NODE_BANDPASS:
        case CHIRP_NODE_MULSPEC:
        case CHIRP_NODE_CONJ:
            /* Handled by the fused-pipeline compiler, not the IR emitter. */
            return;

        case CHIRP_NODE_FFT: {
            size_t n = 64;
            chirp_node *size_node = chirp_node_get_kwarg(node, "size");
            if (size_node && size_node->type == CHIRP_NODE_LITERAL) {
                n = (size_t)size_node->value;
            } else if (node->n_children > 0 && node->children[0]->type == CHIRP_NODE_LITERAL) {
                n = (size_t)node->children[0]->value;
            }
            if (n == 0) n = 64;
            chirp_node *lay_node = chirp_node_get_kwarg(node, "layout");
            if (lay_node) {
                faf_layout lay;
                if (chirp_parse_layout(lay_node, &lay) != 0) {
                    faf_set_error("Chirp: unknown :layout for fft");
                    return;
                }
                t->cfg.layout = lay;
            }
            int inverse = (node->value != 0);
            t->type = inverse ? FAF_TRANSFORM_IFFT : FAF_TRANSFORM_FFT;
            if (inverse) t->flags |= FAF_FLAG_INVERSE;
            if (n > t->n) t->n = n;
            if (!faf_is_5_smooth(n)) {
                faf_set_error("Chirp: FFT size must be 5-smooth, got %zu; "
                              "nearest is %zu", n,
                              faf_get_recommended_size(FAF_TRANSFORM_FFT, n));
                return;
            }
            chirp_ensure_fft_twiddles(t, t->n, inverse);

            faf_inst br = {0};
            br.packed = FAF_BITREV;
            chirp_emit_inst(t, br, inst_count);

            if (faf_is_power_of_2(n)) {
                size_t bits = chirp_log2_size(n);
                for (size_t stage = 0; stage < bits; stage++) {
                    faf_inst st = {0};
                    size_t group_size = 2u << stage;
                    st.packed = FAF_FFT_STAGE;
                    st.a0 = (uint32_t)group_size;
                    st.a1 = 1;
                    st.a2 = (uint32_t)(n / group_size);
                    chirp_emit_inst(t, st, inst_count);
                }
            } else {
                int fac[16], nf = 0;
                faf_factor_5smooth(n, fac, &nf);
                size_t group = 1;
                for (int k = 0; k < nf; k++) {
                    group *= (size_t)fac[k];
                    faf_inst st = {0};
                    st.packed = FAF_FFT_STAGE;
                    st.a0 = (uint32_t)group;
                    st.a1 = (fac[k] == 2) ? 1u : (uint32_t)fac[k];
                    st.a2 = (uint32_t)(n / group);
                    chirp_emit_inst(t, st, inst_count);
                }
            }
            return;
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
            /* Known family pairs expand to a real DWT level. Custom names
             * are not encoded as LIFT_PRED(builtin_index) — that ABI is a
             * register+coefficient primitive, not a function call. */
            chirp_node *predict_node = chirp_node_get_kwarg(node, "predict");
            chirp_node *update_node = chirp_node_get_kwarg(node, "update");
            faf_wavelet_family fam = FAF_WAVELET_HAAR;
            int got = 0;
            if (chirp_parse_family(predict_node, &fam) == 0) got = 1;
            else if (chirp_parse_family(update_node, &fam) == 0) got = 1;
            else {
                const char *pn = chirp_node_name(predict_node);
                const char *un = chirp_node_name(update_node);
                if (pn && (strstr(pn, "haar") || strstr(pn, "linear") ||
                           strstr(pn, "cubic"))) {
                    if (strstr(pn, "cubic") || (un && strstr(un, "cubic")))
                        fam = FAF_WAVELET_CDF97;
                    else if (strstr(pn, "linear") || (un && strstr(un, "linear")))
                        fam = FAF_WAVELET_CDF53;
                    else
                        fam = FAF_WAVELET_HAAR;
                    got = 1;
                }
            }
            if (!got) {
                faf_set_error("Chirp: lift requires a known family (haar/d4/cdf53/cdf97/sym4)");
                return;
            }
            t->family = fam;
            if (t->levels == 0) t->levels = 1;
            t->type = (fam == FAF_WAVELET_HAAR) ? FAF_TRANSFORM_HAAR :
                      (fam == FAF_WAVELET_D4) ? FAF_TRANSFORM_DAUBECHIES4 :
                      (fam == FAF_WAVELET_CDF53) ? FAF_TRANSFORM_CDF53 :
                      (fam == FAF_WAVELET_CDF97) ? FAF_TRANSFORM_CDF97 :
                      FAF_TRANSFORM_SYM4;
            size_t n = t->n ? t->n : 64;
            chirp_emit_dwt_stages(t, inst_count, fam, n, 1, 0);
            return;
        }

        case CHIRP_NODE_DWT:
        case CHIRP_NODE_IDWT: {
            int inverse = (node->type == CHIRP_NODE_IDWT);
            faf_wavelet_family fam = FAF_WAVELET_HAAR;
            chirp_node *fam_node = chirp_node_get_kwarg(node, "family");
            if (chirp_parse_family(fam_node, &fam) != 0 && fam_node && fam_node->sym) {
                faf_set_error("Chirp: unknown wavelet family '%s'", fam_node->sym);
                return;
            }
            size_t n = t->n ? t->n : 64;
            chirp_node *size_node = chirp_node_get_kwarg(node, "size");
            if (size_node) n = (size_t)chirp_node_int(size_node, (int)n);
            if (n == 0) n = 64;
            if (n > t->n) t->n = n;

            size_t max_levels = chirp_log2_size(n);
            size_t levels = max_levels;
            chirp_node *lv_node = chirp_node_get_kwarg(node, "levels");
            if (lv_node) {
                int lv = chirp_node_int(lv_node, 0);
                if (lv > 0) levels = (size_t)lv;
            }
            if (levels > max_levels) levels = max_levels;
            if (levels == 0) levels = max_levels;

            t->family = fam;
            t->levels = levels;
            t->type = (fam == FAF_WAVELET_HAAR) ? FAF_TRANSFORM_HAAR :
                      (fam == FAF_WAVELET_D4) ? FAF_TRANSFORM_DAUBECHIES4 :
                      (fam == FAF_WAVELET_CDF53) ? FAF_TRANSFORM_CDF53 :
                      (fam == FAF_WAVELET_CDF97) ? FAF_TRANSFORM_CDF97 :
                      FAF_TRANSFORM_SYM4;
            if (inverse) t->flags |= FAF_FLAG_INVERSE;
            chirp_emit_dwt_stages(t, inst_count, fam, n, levels, inverse);
            return;
        }

        case CHIRP_NODE_THRESHOLD: {
            uint32_t mode = FAF_THRESH_SOFT;
            chirp_node *mode_node = chirp_node_get_kwarg(node, "mode");
            const char *mname = chirp_node_name(mode_node);
            if (mname && strcmp(mname, "hard") == 0) mode = FAF_THRESH_HARD;
            double lambda = 0.1;
            chirp_node *lam_node = chirp_node_get_kwarg(node, "lambda");
            if (lam_node) lambda = chirp_node_float(lam_node, 0.1);
            size_t skip = 0;
            if (t->levels > 0 && t->n > 0)
                skip = t->n >> t->levels;
            union { uint32_t u; float f; } bits;
            bits.f = (float)lambda;
            inst.packed = FAF_THRESHOLD;
            inst.a0 = mode;
            inst.a1 = (uint32_t)skip;
            inst.a2 = bits.u;
            break;
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
                faf_wavelet_family fam;
                if (faf_wavelet_from_name(fn_name, &fam) == 0) {
                    size_t n = t->n ? t->n : 64;
                    size_t levels = t->levels ? t->levels : 1;
                    chirp_emit_dwt_stages(t, inst_count, fam, n, levels, 0);
                    return;
                }
                inst.packed = FAF_CALL_BUILTIN;
                inst.a0 = chirp_lookup_builtin(fn_name);
                if ((int)inst.a0 < 0) {
                    faf_set_error("Chirp: unknown builtin '%s'", fn_name);
                    fprintf(stderr, "Chirp: unknown builtin '%s'\n", fn_name);
                    inst.a0 = 0;
                }
            }
            break;
        }

        case CHIRP_NODE_LIST: {
            /* A parenthesized expression whose operator is a registered builtin,
             * e.g. (sin) or (sqrt). For now we support zero-argument calls; any
             * children are ignored for scalar builtins. */
            if (node->sym) {
                faf_wavelet_family fam;
                if (faf_wavelet_from_name(node->sym, &fam) == 0) {
                    size_t n = t->n ? t->n : 64;
                    size_t levels = t->levels ? t->levels : 1;
                    chirp_emit_dwt_stages(t, inst_count, fam, n, levels, 0);
                    return;
                }
                int idx = chirp_lookup_builtin(node->sym);
                if (idx >= 0) {
                    inst.packed = FAF_CALL_BUILTIN;
                    inst.a0 = (uint32_t)idx;
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

static int chirp_ast_has_rfft(const chirp_node *n) {
    if (!n) return 0;
    if (n->type == CHIRP_NODE_RFFT || n->type == CHIRP_NODE_IRFFT)
        return 1;
    for (int i = 0; i < n->n_children; i++) {
        if (chirp_ast_has_rfft(n->children[i]))
            return 1;
    }
    return 0;
}

static faf_transform *chirp_compile_rfft_node(chirp_node *node) {
    faf_config c = faf_config_init(0);
    chirp_node *size_node = chirp_node_get_kwarg(node, "size");
    if (size_node && size_node->type == CHIRP_NODE_LITERAL)
        c.n = (size_t)size_node->value;
    else if (node->n_children > 0 &&
             node->children[0]->type == CHIRP_NODE_LITERAL)
        c.n = (size_t)node->children[0]->value;
    if (c.n == 0)
        c.n = 64;
    if (node->type == CHIRP_NODE_IRFFT)
        c.dir = FAF_DIR_INVERSE;

    chirp_node *norm_node = chirp_node_get_kwarg(node, "norm");
    if (norm_node && chirp_parse_norm(norm_node, &c.norm) != 0) {
        faf_set_error("Chirp: unknown :norm for rfft");
        return NULL;
    }
    chirp_node *lay_node = chirp_node_get_kwarg(node, "layout");
    if (lay_node && chirp_parse_layout(lay_node, &c.layout) != 0) {
        faf_set_error("Chirp: unknown :layout for rfft");
        return NULL;
    }
    return faf_create_rfft(&c);
}

static faf_transform *chirp_compile_cwt_node(chirp_node *node) {
    size_t n = 0;
    chirp_node *n_node = chirp_node_get_kwarg(node, "n");
    if (!n_node) n_node = chirp_node_get_kwarg(node, "size");
    if (n_node) n = (size_t)chirp_node_int(n_node, 0);
    if (n == 0) {
        faf_set_error("Chirp: (cwt) requires :n");
        return NULL;
    }

    faf_cwt_config cfg = faf_cwt_config_init(n);

    chirp_node *fs_node = chirp_node_get_kwarg(node, "fs");
    if (fs_node) cfg.fs = chirp_node_float(fs_node, cfg.fs);

    chirp_node *wav_node = chirp_node_get_kwarg(node, "wavelet");
    if (wav_node) {
        const char *wname = chirp_node_name(wav_node);
        if (wname) {
            if (strcmp(wname, "morlet") == 0)       cfg.wavelet = FAF_CWT_WAVELET_MORLET;
            else if (strcmp(wname, "morse") == 0)    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
            else if (strcmp(wname, "bump") == 0)     cfg.wavelet = FAF_CWT_WAVELET_BUMP;
            else if (strcmp(wname, "shannon") == 0)  cfg.wavelet = FAF_CWT_WAVELET_SHANNON;
            else if (strcmp(wname, "meyer") == 0)    cfg.wavelet = FAF_CWT_WAVELET_MEYER;
            else {
                faf_set_error("Chirp: unknown CWT wavelet '%s'", wname);
                return NULL;
            }
        }
    }

    chirp_node *v_node = chirp_node_get_kwarg(node, "voices");
    if (v_node) cfg.voices = (size_t)chirp_node_int(v_node, (int)cfg.voices);

    chirp_node *fmin_node = chirp_node_get_kwarg(node, "fmin");
    if (fmin_node) cfg.fmin = chirp_node_float(fmin_node, cfg.fmin);

    chirp_node *fmax_node = chirp_node_get_kwarg(node, "fmax");
    if (fmax_node) cfg.fmax = chirp_node_float(fmax_node, cfg.fmax);

    chirp_node *gamma_node = chirp_node_get_kwarg(node, "gamma");
    if (gamma_node) cfg.morse_gamma = chirp_node_float(gamma_node, cfg.morse_gamma);

    chirp_node *beta_node = chirp_node_get_kwarg(node, "beta");
    if (beta_node) cfg.morse_beta = chirp_node_float(beta_node, cfg.morse_beta);

    chirp_node *mu_node = chirp_node_get_kwarg(node, "mu");
    if (mu_node) cfg.morlet_mu = chirp_node_float(mu_node, cfg.morlet_mu);

    chirp_node *norm_node = chirp_node_get_kwarg(node, "norm");
    if (norm_node) {
        const char *nname = chirp_node_name(norm_node);
        if (nname) {
            if (strcmp(nname, "l1") == 0)            cfg.norm = FAF_CWT_NORM_L1;
            else if (strcmp(nname, "l2") == 0)       cfg.norm = FAF_CWT_NORM_L2;
            else if (strcmp(nname, "bandpass") == 0)  cfg.norm = FAF_CWT_NORM_BANDPASS;
            else {
                faf_set_error("Chirp: unknown CWT norm '%s'", nname);
                return NULL;
            }
        }
    }

    chirp_node *prec_node = chirp_node_get_kwarg(node, "precision");
    if (prec_node) {
        const char *pname = chirp_node_name(prec_node);
        if (pname) {
            if (strcmp(pname, "f32") == 0)       cfg.precision = FAF_PREC_FP32;
            else if (strcmp(pname, "f64") == 0)  cfg.precision = FAF_PREC_FP64;
            else {
                faf_set_error("Chirp: CWT only supports f32/f64 precision");
                return NULL;
            }
        }
    }

    chirp_node *lp_node = chirp_node_get_kwarg(node, "lowpass");
    if (lp_node) {
        const char *lpname = chirp_node_name(lp_node);
        if (lpname && strcmp(lpname, "off") == 0)
            cfg.flags &= ~FAF_CWT_FLAG_INCLUDE_LOWPASS;
        else
            cfg.flags |= FAF_CWT_FLAG_INCLUDE_LOWPASS;
    }

    chirp_node *ut_node = chirp_node_get_kwarg(node, "allow-untiled");
    if (ut_node) cfg.flags |= FAF_CWT_FLAG_ALLOW_UNTILED;

    if (node->type == CHIRP_NODE_ICWT) {
        faf_cwt_inverse_kind kind = FAF_CWT_INV_DUAL;
        chirp_node *inv_node = chirp_node_get_kwarg(node, "inverse");
        if (inv_node) {
            const char *iname = chirp_node_name(inv_node);
            if (iname && strcmp(iname, "l1") == 0) kind = FAF_CWT_INV_L1;
        }
        return faf_create_icwt(&cfg, kind);
    }

    return faf_create_cwt(&cfg);
}

static int chirp_ast_has_cwt(const chirp_node *n) {
    if (!n) return 0;
    if (n->type == CHIRP_NODE_CWT || n->type == CHIRP_NODE_ICWT)
        return 1;
    for (int i = 0; i < n->n_children; i++) {
        if (chirp_ast_has_cwt(n->children[i]))
            return 1;
    }
    return 0;
}

static int chirp_node_is_spectral_op(chirp_node *n) {
    if (!n) return 0;
    if (n->type == CHIRP_NODE_SPECTRAL || n->type == CHIRP_NODE_BANDPASS ||
        n->type == CHIRP_NODE_MULSPEC || n->type == CHIRP_NODE_CONJ)
        return 1;
    if (n->type == CHIRP_NODE_CUSTOM && n->sym) {
        int idx = chirp_lookup_builtin(n->sym);
        int k = chirp_builtin_kind(idx);
        return k == CHIRP_KIND_SPECTRAL || k == CHIRP_KIND_HERMITIAN ||
               k == CHIRP_KIND_COMPLEX || k == CHIRP_KIND_VECTOR;
    }
    if ((n->type == CHIRP_NODE_LITERAL || n->type == CHIRP_NODE_LIST) &&
        n->sym) {
        if (strcmp(n->sym, "conj") == 0 ||
            strcmp(n->sym, "mul-spectrum") == 0)
            return 1;
        int idx = chirp_lookup_builtin(n->sym);
        if (idx >= 0) {
            int k = chirp_builtin_kind(idx);
            return k == CHIRP_KIND_SPECTRAL || k == CHIRP_KIND_HERMITIAN ||
                   k == CHIRP_KIND_COMPLEX || k == CHIRP_KIND_VECTOR;
        }
    }
    return 0;
}

static int chirp_emit_pipe_op(chirp_node *n, faf_inst *out) {
    memset(out, 0, sizeof(*out));
    out->packed = FAF_CALL_BUILTIN;
    if (n->type == CHIRP_NODE_CONJ ||
        (n->sym && strcmp(n->sym, "conj") == 0)) {
        out->a0 = CHIRP_OP_CONJ;
        return 0;
    }
    if (n->type == CHIRP_NODE_MULSPEC ||
        (n->sym && strcmp(n->sym, "mul-spectrum") == 0)) {
        out->a0 = CHIRP_OP_MUL;
        return 0;
    }
    if (n->type == CHIRP_NODE_BANDPASS) {
        chirp_node *lo = chirp_node_get_kwarg(n, "lo");
        chirp_node *hi = chirp_node_get_kwarg(n, "hi");
        out->a0 = CHIRP_OP_BANDPASS;
        out->a1 = (uint32_t)chirp_node_int(lo, 0);
        out->a2 = (uint32_t)chirp_node_int(hi, 0);
        return 0;
    }
    const char *name = NULL;
    if (n->type == CHIRP_NODE_SPECTRAL) {
        chirp_node *nm = chirp_node_get_kwarg(n, "name");
        if (nm) name = chirp_node_name(nm);
        else if (n->n_children > 0) name = chirp_node_name(n->children[0]);
        else name = n->sym;
    } else {
        name = chirp_node_name(n);
    }
    if (!name) {
        faf_set_error("Chirp: spectral form needs a registered name");
        return -1;
    }
    int idx = chirp_lookup_builtin(name);
    if (idx < 0) {
        faf_set_error("Chirp: unknown spectral builtin '%s'", name);
        return -1;
    }
    out->a0 = (uint32_t)idx;
    return 0;
}

static faf_transform *chirp_compile_spectral_pipeline(chirp_node *ast) {
    chirp_node **kids = ast->children;
    int nch = ast->n_children;
    if (nch < 1 || kids[0]->type != CHIRP_NODE_RFFT)
        return NULL;

    int has_irfft = (nch >= 2 && kids[nch - 1]->type == CHIRP_NODE_IRFFT);
    int mid0 = 1;
    int mid1 = has_irfft ? nch - 1 : nch;
    for (int i = mid0; i < mid1; i++) {
        if (!chirp_node_is_spectral_op(kids[i])) {
            faf_set_error("Chirp: fused rfft pipeline only accepts "
                          "spectral/conj/mul-spectrum/bandpass in the middle");
            return NULL;
        }
    }

    faf_transform *fwd = chirp_compile_rfft_node(kids[0]);
    if (!fwd) return NULL;

    faf_transform *inv = NULL;
    if (has_irfft) {
        chirp_node *inode = kids[nch - 1];
        chirp_node *sz = chirp_node_get_kwarg(inode, "size");
        if (!sz && inode->n_children == 0) {
            /* inherit size/norm/layout from the forward rfft */
            faf_config ic = faf_config_inverse(fwd->cfg);
            inv = faf_create_rfft(&ic);
        } else {
            inv = chirp_compile_rfft_node(inode);
        }
        if (!inv) {
            faf_destroy_transform(fwd);
            return NULL;
        }
    }

    int nops = mid1 - mid0;
    faf_transform *t = calloc(1, sizeof(faf_transform));
    if (!t) {
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
        return NULL;
    }
    t->type = FAF_TRANSFORM_PIPELINE;
    t->n = fwd->n;
    t->precision = fwd->precision;
    t->inner = fwd;
    t->inner_inv = inv;
    t->cfg = fwd->cfg;
    /* Time I/O is REAL when irfft is present. Spectrum stays on fwd. */
    t->cfg.layout = has_irfft ? FAF_LAYOUT_REAL : fwd->cfg.layout;
    if (ast->type == CHIRP_NODE_PIPELINE) {
        chirp_node *play = chirp_node_get_kwarg(ast, "layout");
        if (play) {
            faf_layout lay;
            if (chirp_parse_layout(play, &lay) != 0) {
                faf_set_error("Chirp: unknown :layout for pipeline");
                faf_destroy_transform(t);
                return NULL;
            }
            /* :layout on a full round-trip is the time-domain layout. */
            if (has_irfft && lay != FAF_LAYOUT_DEFAULT)
                t->cfg.layout = (lay == FAF_LAYOUT_INTERLEAVED)
                    ? FAF_LAYOUT_REAL : lay;
            else if (!has_irfft && lay != FAF_LAYOUT_DEFAULT)
                t->cfg.layout = lay;
        }
    }
    t->cfg.dir = has_irfft ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    t->flags = FAF_FLAG_REAL;

    if (nops > 0) {
        t->code = calloc((size_t)nops, sizeof(faf_inst));
        if (!t->code) {
            faf_destroy_transform(t);
            return NULL;
        }
        t->n_inst = (size_t)nops;
        for (int i = 0; i < nops; i++) {
            if (chirp_emit_pipe_op(kids[mid0 + i], &t->code[i]) != 0) {
                faf_destroy_transform(t);
                return NULL;
            }
        }
    }

    size_t nbins = t->n / 2 + 1;
    size_t elem = (t->precision == FAF_PREC_FP64) ? sizeof(double)
                                                  : sizeof(float);
    size_t bytes = ((nbins * 2 * elem) + 63u) & ~(size_t)63u;
    t->scratch = aligned_alloc(64, bytes);
    if (!t->scratch) {
        faf_set_error("Chirp: failed to allocate pipeline scratch");
        faf_destroy_transform(t);
        return NULL;
    }
    t->scratch_size = bytes;
    memset(t->scratch, 0, bytes);
    return t;
}

/* Public API: Compile a Chirp program */
faf_transform* chirp_compile(const char *source) {
    if (!source) return NULL;
    
    chirp_init_builtins();
    
    /* Parse the source */
    chirp_lexer lex = chirp_lexer_new(source);
    chirp_node *ast = chirp_parse_expr(&lex);
    
    if (!ast) {
        faf_set_error("Chirp: failed to parse program");
        fprintf(stderr, "Chirp: failed to parse program\n");
        return NULL;
    }

    /* Standalone (rfft)/(irfft), or a fused R2C → spectral C → C2R pipeline. */
    chirp_node *rfft_node = NULL;
    if (ast->type == CHIRP_NODE_RFFT || ast->type == CHIRP_NODE_IRFFT)
        rfft_node = ast;
    else if (ast->type == CHIRP_NODE_PIPELINE && ast->n_children == 1 &&
             (ast->children[0]->type == CHIRP_NODE_RFFT ||
              ast->children[0]->type == CHIRP_NODE_IRFFT))
        rfft_node = ast->children[0];
    if (rfft_node) {
        faf_transform *rt = chirp_compile_rfft_node(rfft_node);
        chirp_node_free(ast);
        return rt;
    }
    if (ast->type == CHIRP_NODE_PIPELINE && ast->n_children >= 1 &&
        ast->children[0]->type == CHIRP_NODE_RFFT) {
        faf_transform *pt = chirp_compile_spectral_pipeline(ast);
        chirp_node_free(ast);
        return pt;
    }
    if (chirp_ast_has_rfft(ast)) {
        faf_set_error("Chirp: (rfft)/(irfft) must lead a fused pipeline "
                      "(rfft … irfft); they cannot mix with FFT IR stages");
        chirp_node_free(ast);
        return NULL;
    }

    /* Standalone (cwt)/(icwt) — not composable in pipelines. */
    if (ast->type == CHIRP_NODE_CWT || ast->type == CHIRP_NODE_ICWT) {
        faf_transform *ct = chirp_compile_cwt_node(ast);
        chirp_node_free(ast);
        return ct;
    }
    if (chirp_ast_has_cwt(ast)) {
        faf_set_error("Chirp: (cwt)/(icwt) must be standalone; "
                      "they cannot appear inside a pipeline");
        chirp_node_free(ast);
        return NULL;
    }

    /* Create transform */
    faf_transform *t = calloc(1, sizeof(faf_transform));
    t->precision = FAF_PREC_FP32;
    t->n = 0; /* filled in by (fft/:dwt :size) or defaulted after the first pass */
    
    /* First pass: count instructions and resolve the transform size from
     * high-level forms like (fft :size N). */
    int ast_inst_count = 0;
    chirp_compile_node_emit(ast, t, &ast_inst_count);
    if (t->n == 0) t->n = 64;
    
    /* Wrap the user pipeline with LOAD/STORE so it operates on real input and
     * output data. The AST may have updated t->n based on (fft :size N). */
    size_t io_count = t->n;
    int total_inst = ast_inst_count + 2 * (int)io_count + 1; /* +1 for END */
    
    t->n_inst = total_inst;
    t->code = calloc(t->n_inst, sizeof(faf_inst));
    if (!t->code) {
        faf_set_error("Chirp: failed to allocate instruction array");
        fprintf(stderr, "Chirp: failed to allocate instruction array\n");
        chirp_node_free(ast);
        free(t);
        return NULL;
    }
    
    /* Second pass: generate LOAD, user pipeline, STORE, END */
    int inst_count = 0;
    
    for (size_t i = 0; i < io_count; i++) {
        faf_inst load = {0};
        load.packed = FAF_LOAD;
        load.a0 = (uint32_t)i;
        load.a1 = (uint32_t)i;
        t->code[inst_count++] = load;
    }
    
    chirp_compile_node_emit(ast, t, &inst_count);
    
    for (size_t i = 0; i < io_count; i++) {
        faf_inst store = {0};
        store.packed = FAF_STORE;
        store.a0 = (uint32_t)i;
        store.a1 = (uint32_t)i;
        t->code[inst_count++] = store;
    }
    
    /* Add END opcode */
    faf_inst end_inst = {0};
    end_inst.packed = FAF_END;
    t->code[inst_count] = end_inst;
    
    /* Cleanup AST */
    chirp_node_free(ast);

    /* Resolved config so faf_execute / create_inverse work on Chirp programs.
     * Default layout stays interleaved so existing faf_execute_f32 helpers
     * keep working; (fft :layout split) overrides during emit. */
    faf_layout lay = t->cfg.layout;
    t->cfg = faf_config_init(t->n);
    t->cfg.precision = t->precision;
    t->cfg.layout = (lay == FAF_LAYOUT_DEFAULT) ? FAF_LAYOUT_INTERLEAVED : lay;
    t->cfg.norm = FAF_NORM_NONE;
    if (t->flags & FAF_FLAG_INVERSE)
        t->cfg.dir = FAF_DIR_INVERSE;
    t->cfg.family = t->family;
    t->cfg.levels = t->levels;

    return t;
}
