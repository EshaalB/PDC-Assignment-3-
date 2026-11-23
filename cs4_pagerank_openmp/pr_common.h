/* pr_common.h
 * Shared graph representation, graph construction/loading, options, timing,
 * and serial PageRank reference for Case Study 4.
 *
 * This project intentionally has NO separate graph.h dependency.
 */
#ifndef PR_COMMON_H
#define PR_COMMON_H

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Compressed Sparse Row representation. */
typedef struct {
    uint32_t n;
    uint64_t m;
    uint64_t *off;       /* n + 1 offsets */
    uint32_t *adj;       /* m neighbour IDs */
} CSR;

typedef struct {
    uint32_t n;
    uint64_t m;          /* directed edges after removing loops/duplicates */
    CSR out;             /* outgoing neighbours */
    CSR in;              /* incoming neighbours */
} Graph;

typedef struct {
    uint32_t src, dst;
} PREdge;

typedef struct {
    int         scale, ef, perm;
    uint64_t    seed;
    const char *file;
    int         reps, maxit, csv;
    double      eps, damping;
    const char *degdump;
} Opts;

/* Small deterministic SplitMix64 generator. */
static uint64_t pr_rng_next(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t pr_rng_bounded(uint64_t *state, uint64_t bound)
{
    /* Rejection sampling avoids modulo bias. */
    if (bound <= 1) return 0;
    uint64_t threshold = (uint64_t)(-bound) % bound;
    for (;;) {
        uint64_t x = pr_rng_next(state);
        if (x >= threshold) return x % bound;
    }
}

static int pr_edge_cmp_src(const void *a, const void *b)
{
    const PREdge *x = (const PREdge *)a, *y = (const PREdge *)b;
    if (x->src != y->src) return (x->src > y->src) - (x->src < y->src);
    return (x->dst > y->dst) - (x->dst < y->dst);
}

static int pr_edge_cmp_dst(const void *a, const void *b)
{
    const PREdge *x = (const PREdge *)a, *y = (const PREdge *)b;
    if (x->dst != y->dst) return (x->dst > y->dst) - (x->dst < y->dst);
    return (x->src > y->src) - (x->src < y->src);
}

static void *pr_xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *pr_xcalloc(size_t count, size_t bytes)
{
    if (bytes && count > SIZE_MAX / bytes) {
        fprintf(stderr, "allocation size overflow\n");
        exit(EXIT_FAILURE);
    }
    void *p = calloc(count ? count : 1, bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", count * bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

/* Generate a directed R-MAT graph. Raw duplicates and self-loops are removed
 * later by build_graph(). The generator is deterministic for a given seed. */
static void gen_rmat(int scale, int edgefactor, uint64_t seed, int permute,
                     uint32_t *n_out, uint64_t *m_out,
                     uint32_t **src_out, uint32_t **dst_out)
{
    if (scale < 1 || scale > 31 || edgefactor < 1) {
        fprintf(stderr, "invalid R-MAT parameters: scale must be 1..31 and ef >= 1\n");
        exit(EXIT_FAILURE);
    }

    uint32_t n = UINT32_C(1) << scale;
    if ((uint64_t)n > UINT64_MAX / (uint64_t)edgefactor) {
        fprintf(stderr, "R-MAT edge count overflow\n");
        exit(EXIT_FAILURE);
    }
    uint64_t m = (uint64_t)n * (uint64_t)edgefactor;
    if (m > SIZE_MAX / sizeof(uint32_t)) {
        fprintf(stderr, "R-MAT graph is too large for this platform\n");
        exit(EXIT_FAILURE);
    }

    uint32_t *src = (uint32_t *)pr_xmalloc((size_t)m * sizeof(uint32_t));
    uint32_t *dst = (uint32_t *)pr_xmalloc((size_t)m * sizeof(uint32_t));
    uint32_t *perm = NULL;
    uint64_t rng = seed;

    if (permute) {
        perm = (uint32_t *)pr_xmalloc((size_t)n * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; ++i) perm[i] = i;
        for (uint64_t i = n - 1; i > 0; --i) {
            uint64_t j = pr_rng_bounded(&rng, i + 1);
            uint32_t tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
    }

    for (uint64_t e = 0; e < m; ++e) {
        uint32_t u = 0, v = 0;
        for (int bit = scale - 1; bit >= 0; --bit) {
            /* R-MAT quadrant probabilities: A=.57, B=.19, C=.19, D=.05. */
            uint64_t r = pr_rng_next(&rng) >> 11;
            double x = (double)r * (1.0 / 9007199254740992.0);
            int row = 0, col = 0;
            if (x >= 0.57) {
                if (x < 0.76) col = 1;
                else if (x < 0.95) row = 1;
                else row = col = 1;
            }
            u |= (uint32_t)row << bit;
            v |= (uint32_t)col << bit;
        }
        src[e] = permute ? perm[u] : u;
        dst[e] = permute ? perm[v] : v;
    }

    free(perm);
    *n_out = n;
    *m_out = m;
    *src_out = src;
    *dst_out = dst;
}

/* Read a SNAP-style directed edge list ("source destination", # comments).
 * Vertex IDs are retained as-is, so n = max_vertex_id + 1. */
static int load_snap(const char *filename, uint32_t *n_out, uint64_t *m_out,
                     uint32_t **src_out, uint32_t **dst_out)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "cannot open SNAP graph '%s': %s\n", filename, strerror(errno));
        return -1;
    }

    size_t cap = 1024;
    uint64_t count = 0;
    uint32_t *src = (uint32_t *)pr_xmalloc(cap * sizeof(uint32_t));
    uint32_t *dst = (uint32_t *)pr_xmalloc(cap * sizeof(uint32_t));
    uint32_t max_id = 0;
    int have_vertex = 0;
    char *line = NULL;
    size_t linecap = 0;
    ssize_t len;

    while ((len = getline(&line, &linecap, fp)) != -1) {
        (void)len;
        char *s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == '\0' || *s == '\n' || *s == '#' || *s == '%') continue;

        unsigned long long a, b;
        char extra;
        int got = sscanf(s, "%llu %llu %c", &a, &b, &extra);
        if (got < 2) continue;
        if (a > UINT32_MAX || b > UINT32_MAX) {
            fprintf(stderr, "SNAP vertex ID exceeds uint32 range in '%s'\n", filename);
            free(line); free(src); free(dst); fclose(fp);
            return -1;
        }
        if (count == cap) {
            if (cap > SIZE_MAX / (2 * sizeof(uint32_t))) {
                fprintf(stderr, "SNAP edge list is too large\n");
                free(line); free(src); free(dst); fclose(fp);
                return -1;
            }
            cap *= 2;
            src = (uint32_t *)realloc(src, cap * sizeof(uint32_t));
            dst = (uint32_t *)realloc(dst, cap * sizeof(uint32_t));
            if (!src || !dst) {
                fprintf(stderr, "out of memory growing SNAP edge list\n");
                free(line); free(src); free(dst); fclose(fp);
                return -1;
            }
        }
        src[count] = (uint32_t)a;
        dst[count] = (uint32_t)b;
        if ((uint32_t)a > max_id) max_id = (uint32_t)a;
        if ((uint32_t)b > max_id) max_id = (uint32_t)b;
        ++count;
        have_vertex = 1;
    }

    free(line);
    fclose(fp);
    if (!have_vertex) {
        fprintf(stderr, "SNAP graph '%s' contains no edges\n", filename);
        free(src); free(dst);
        return -1;
    }
    if (max_id == UINT32_MAX) {
        fprintf(stderr, "SNAP graph vertex range cannot be represented\n");
        free(src); free(dst);
        return -1;
    }

    *n_out = max_id + 1;
    *m_out = count;
    *src_out = src;
    *dst_out = dst;
    return 0;
}

/* Construct outgoing and incoming CSR arrays, removing self-loops and
 * duplicate directed edges. */
static void build_graph(uint32_t n, uint64_t raw_m,
                        const uint32_t *src, const uint32_t *dst, Graph *g)
{
    if (n == 0) {
        fprintf(stderr, "cannot build a graph with zero vertices\n");
        exit(EXIT_FAILURE);
    }
    if (raw_m > SIZE_MAX / sizeof(PREdge)) {
        fprintf(stderr, "edge list too large for this platform\n");
        exit(EXIT_FAILURE);
    }

    PREdge *edges = (PREdge *)pr_xmalloc((size_t)raw_m * sizeof(PREdge));
    uint64_t kept = 0;
    for (uint64_t i = 0; i < raw_m; ++i) {
        if (src[i] >= n || dst[i] >= n) {
            fprintf(stderr, "edge endpoint outside graph vertex range\n");
            free(edges);
            exit(EXIT_FAILURE);
        }
        if (src[i] == dst[i]) continue;
        edges[kept].src = src[i];
        edges[kept].dst = dst[i];
        ++kept;
    }

    qsort(edges, (size_t)kept, sizeof(PREdge), pr_edge_cmp_src);
    uint64_t unique = 0;
    for (uint64_t i = 0; i < kept; ++i) {
        if (unique == 0 ||
            edges[i].src != edges[unique - 1].src ||
            edges[i].dst != edges[unique - 1].dst) {
            edges[unique++] = edges[i];
        }
    }

    g->n = n;
    g->m = unique;
    g->out.n = g->in.n = n;
    g->out.m = g->in.m = unique;
    g->out.off = (uint64_t *)pr_xcalloc((size_t)n + 1, sizeof(uint64_t));
    g->in.off  = (uint64_t *)pr_xcalloc((size_t)n + 1, sizeof(uint64_t));
    g->out.adj = (uint32_t *)pr_xmalloc((size_t)unique * sizeof(uint32_t));
    g->in.adj  = (uint32_t *)pr_xmalloc((size_t)unique * sizeof(uint32_t));

    for (uint64_t i = 0; i < unique; ++i) {
        ++g->out.off[(uint64_t)edges[i].src + 1];
        ++g->in.off[(uint64_t)edges[i].dst + 1];
    }
    for (uint32_t v = 0; v < n; ++v) {
        g->out.off[v + 1] += g->out.off[v];
        g->in.off[v + 1] += g->in.off[v];
    }

    /* Edges are already sorted by source, so outgoing adjacency is ordered. */
    for (uint64_t i = 0; i < unique; ++i) g->out.adj[i] = edges[i].dst;

    /* Sort by destination to build incoming adjacency in contiguous CSR order. */
    qsort(edges, (size_t)unique, sizeof(PREdge), pr_edge_cmp_dst);
    for (uint64_t i = 0; i < unique; ++i) g->in.adj[i] = edges[i].src;
    free(edges);
}

static uint32_t out_degree(const Graph *g, uint32_t u)
{
    return (uint32_t)(g->out.off[u + 1] - g->out.off[u]);
}

static inline void free_graph(Graph *g)
{
    free(g->out.off); free(g->out.adj);
    free(g->in.off); free(g->in.adj);
    memset(g, 0, sizeof(*g));
}

static void print_graph_stats(const Graph *g, const char *label)
{
    uint64_t dangling = 0;
    uint32_t min_out = UINT32_MAX, max_out = 0;
    for (uint32_t v = 0; v < g->n; ++v) {
        uint32_t d = out_degree(g, v);
        if (d == 0) ++dangling;
        if (d < min_out) min_out = d;
        if (d > max_out) max_out = d;
    }
    printf("Graph                 : %s\n", label);
    printf("Vertices (n)          : %" PRIu32 "\n", g->n);
    printf("Edges (m)             : %" PRIu64 " (after removing loops/duplicates)\n", g->m);
    printf("Dangling vertices     : %" PRIu64 " (%.2f%%)\n",
           dangling, 100.0 * (double)dangling / (double)g->n);
    printf("Out-degree min / max  : %" PRIu32 " / %" PRIu32 "\n",
           min_out == UINT32_MAX ? 0 : min_out, max_out);
}

static void dump_degree_hist(const Graph *g, const char *prefix)
{
    char path[1024];
    for (int which = 0; which < 2; ++which) {
        snprintf(path, sizeof(path), "%s_%s.csv", prefix, which ? "in" : "out");
        FILE *fp = fopen(path, "w");
        if (!fp) {
            fprintf(stderr, "cannot write degree histogram '%s': %s\n", path, strerror(errno));
            continue;
        }
        fprintf(fp, "vertex,degree\n");
        const CSR *c = which ? &g->in : &g->out;
        for (uint32_t v = 0; v < g->n; ++v)
            fprintf(fp, "%" PRIu32 ",%" PRIu64 "\n", v, c->off[v + 1] - c->off[v]);
        fclose(fp);
    }
}

/* --------------------------------------------------------------- options -- */
static void opts_default(Opts *o)
{
    o->scale = 20; o->ef = 16; o->perm = 0; o->seed = 1; o->file = NULL;
    o->reps = 3; o->maxit = 200; o->csv = 0;
    o->eps = 1e-9; o->damping = 0.85; o->degdump = NULL;
}

static int parse_common(Opts *o, int argc, char **argv, int *i)
{
    const char *a = argv[*i];
    int has = (*i + 1 < argc);
    if      (!strcmp(a, "-scale") && has) o->scale = atoi(argv[++*i]);
    else if (!strcmp(a, "-ef")    && has) o->ef    = atoi(argv[++*i]);
    else if (!strcmp(a, "-seed")  && has) o->seed  = strtoull(argv[++*i], NULL, 10);
    else if (!strcmp(a, "-f")     && has) o->file  = argv[++(*i)];
    else if (!strcmp(a, "-r")     && has) o->reps  = atoi(argv[++(*i)]);
    else if (!strcmp(a, "-e")     && has) o->eps   = atof(argv[++(*i)]);
    else if (!strcmp(a, "-i")     && has) o->maxit = atoi(argv[++(*i)]);
    else if (!strcmp(a, "-D")     && has) o->degdump = argv[++(*i)];
    else if (!strcmp(a, "-perm"))         o->perm  = 1;
    else if (!strcmp(a, "--csv"))         o->csv   = 1;
    else return 0;
    return 1;
}

#define COMMON_USAGE "[-scale S] [-ef F] [-seed X] [-perm] [-f snap.txt] [-r reps] [-e eps] [-i maxit] [-D prefix] [--csv]"

static void load_graph(const Opts *o, Graph *g)
{
    uint32_t n;
    uint64_t me;
    uint32_t *src, *dst;
    char label[256];
    if (o->file) {
        if (load_snap(o->file, &n, &me, &src, &dst) != 0) exit(EXIT_FAILURE);
        snprintf(label, sizeof label, "SNAP file %s", o->file);
    } else {
        gen_rmat(o->scale, o->ef, o->seed, o->perm, &n, &me, &src, &dst);
        snprintf(label, sizeof label, "R-MAT scale=%d edgefactor=%d%s", o->scale, o->ef,
                 o->perm ? " permuted" : "");
    }
    build_graph(n, me, src, dst, g);
    free(src); free(dst);
    if (!o->csv) print_graph_stats(g, label);
}

/* ------------------------------------------------------------ timing ------ */
static inline double wtime(void) { return omp_get_wtime(); }

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_d(const double *v, int n)
{
    double *c = (double *)pr_xmalloc((size_t)n * sizeof(double));
    memcpy(c, v, (size_t)n * sizeof(double));
    qsort(c, (size_t)n, sizeof(double), cmp_d);
    double m = (n % 2) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

static double min_d(const double *v, int n)
{
    double m = v[0];
    for (int i = 1; i < n; ++i) if (v[i] < m) m = v[i];
    return m;
}

/* -------------------------------------------------- serial PageRank (pull) -- */
static int pr_serial(const Graph *g, double d, double eps, int maxit, double *r_out)
{
    const uint32_t n = g->n;
    double *cur = (double *)pr_xmalloc((size_t)n * sizeof(double));
    double *nxt = (double *)pr_xmalloc((size_t)n * sizeof(double));
    double *contrib = (double *)pr_xmalloc((size_t)n * sizeof(double));
    for (uint32_t v = 0; v < n; ++v) cur[v] = 1.0 / n;

    int it = 0;
    while (it < maxit) {
        double dangling = 0.0;
        for (uint32_t u = 0; u < n; ++u) {
            uint32_t deg = out_degree(g, u);
            if (deg) contrib[u] = cur[u] / deg;
            else { contrib[u] = 0.0; dangling += cur[u]; }
        }
        const double base = (1.0 - d) / n + d * dangling / n;
        double diff = 0.0;
        for (uint32_t v = 0; v < n; ++v) {
            double s = 0.0;
            for (uint64_t e = g->in.off[v]; e < g->in.off[v + 1]; ++e)
                s += contrib[g->in.adj[e]];
            double x = base + d * s;
            diff += fabs(x - cur[v]);
            nxt[v] = x;
        }
        double *tmp = cur; cur = nxt; nxt = tmp;
        ++it;
        if (diff < eps) break;
    }
    memcpy(r_out, cur, (size_t)n * sizeof(double));
    free(cur); free(nxt); free(contrib);
    return it;
}

#endif /* PR_COMMON_H */
