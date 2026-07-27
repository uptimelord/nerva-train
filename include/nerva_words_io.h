/* SPDX-License-Identifier: Apache-2.0
 * Word lexicon sidecar for frozen checkpoints (FluencyWords).
 */
#ifndef NERVA_WORDS_IO_H
#define NERVA_WORDS_IO_H

#include "fluency.h"

#include <stdio.h>
#include <string.h>

#define NERVA_WORDS_MAGIC "NrvWrds1"

static inline int nerva_words_save(const FluencyWords *w, const char *path) {
    FILE *f;
    uint32_t i;
    if (!w || !path)
        return -1;
    f = fopen(path, "wb");
    if (!f)
        return -1;
    if (fwrite(NERVA_WORDS_MAGIC, 1, 8, f) != 8) {
        fclose(f);
        return -1;
    }
    if (fwrite(&w->count, sizeof(w->count), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    for (i = 0; i < w->count && i < FLUENCY_VOCAB; ++i) {
        uint8_t len = (uint8_t)strlen(w->word[i]);
        if (fwrite(&len, 1, 1, f) != 1 || (len && fwrite(w->word[i], 1, len, f) != len)) {
            fclose(f);
            return -1;
        }
        if (fwrite(&w->freq[i], sizeof(w->freq[i]), 1, f) != 1) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

static inline int nerva_words_load(FluencyWords *w, const char *path) {
    FILE *f;
    char magic[8];
    uint32_t n = 0, i;
    if (!w || !path)
        return -1;
    fluency_words_init(w);
    f = fopen(path, "rb");
    if (!f)
        return -1;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, NERVA_WORDS_MAGIC, 8) != 0) {
        fclose(f);
        return -1;
    }
    if (fread(&n, sizeof(n), 1, f) != 1 || n > FLUENCY_VOCAB) {
        fclose(f);
        return -1;
    }
    for (i = 0; i < n; ++i) {
        uint8_t len = 0;
        if (fread(&len, 1, 1, f) != 1 || len > FLUENCY_WORD_LEN) {
            fclose(f);
            return -1;
        }
        memset(w->word[i], 0, sizeof(w->word[i]));
        if (len && fread(w->word[i], 1, len, f) != len) {
            fclose(f);
            return -1;
        }
        if (fread(&w->freq[i], sizeof(w->freq[i]), 1, f) != 1) {
            fclose(f);
            return -1;
        }
    }
    w->count = n;
    fclose(f);
    return 0;
}

#endif
