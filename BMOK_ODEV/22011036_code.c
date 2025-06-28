#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>  // fmax ve fmin fonksiyonlarý için eklendi

#define MAX_LINE 256  // Maksimum satýr uzunluðu

// Laplacian Kernel 1 (Merkez: 4, Çevre: -1)
const double laplacian_kernel1[] = {
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
};

// Laplacian Kernel 2 (Merkez: 8, Çevre: -1)
const double laplacian_kernel2[] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

const double gauss3x3_1_0[] = {
    0.0751136, 0.123841, 0.0751136,
    0.123841,  0.204180, 0.123841,
    0.0751136, 0.123841, 0.0751136
};

const double gauss3x3_2_0[] = {
    0.102059, 0.115349, 0.102059,
    0.115349, 0.130776, 0.115349,
    0.102059, 0.115349, 0.102059
};

const double gauss3x3_4_0[] = {
    0.111111, 0.111111, 0.111111,
    0.111111, 0.111111, 0.111111,
    0.111111, 0.111111, 0.111111
}; // Ortalama filtresine benzer

const double gauss5x5_1_0[] = {
    0.002969, 0.013306, 0.021938, 0.013306, 0.002969,
    0.013306, 0.059634, 0.098320, 0.059634, 0.013306,
    0.021938, 0.098320, 0.162103, 0.098320, 0.021938,
    0.013306, 0.059634, 0.098320, 0.059634, 0.013306,
    0.002969, 0.013306, 0.021938, 0.013306, 0.002969
};

const double gauss5x5_2_0[] = {
    0.01236, 0.02679, 0.03342, 0.02679, 0.01236,
    0.02679, 0.05808, 0.07248, 0.05808, 0.02679,
    0.03342, 0.07248, 0.09052, 0.07248, 0.03342,
    0.02679, 0.05808, 0.07248, 0.05808, 0.02679,
    0.01236, 0.02679, 0.03342, 0.02679, 0.01236
};

const double gauss5x5_4_0[] = {
    0.02988, 0.03852, 0.03999, 0.03852, 0.02988,
    0.03852, 0.04968, 0.05162, 0.04968, 0.03852,
    0.03999, 0.05162, 0.05367, 0.05162, 0.03999,
    0.03852, 0.04968, 0.05162, 0.04968, 0.03852,
    0.02988, 0.03852, 0.03999, 0.03852, 0.02988
};

const double gauss7x7_1_0[] = {
    0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
    0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
    0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
    0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
    0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
    0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
    0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036
};

const double gauss7x7_2_0[] = {
    0.000429, 0.002967, 0.007883, 0.01087, 0.007883, 0.002967, 0.000429,
    0.002967, 0.02052, 0.05453, 0.07523, 0.05453, 0.02052, 0.002967,
    0.007883, 0.05453, 0.14506, 0.20013, 0.14506, 0.05453, 0.007883,
    0.01087,  0.07523, 0.20013, 0.27591, 0.20013, 0.07523, 0.01087,
    0.007883, 0.05453, 0.14506, 0.20013, 0.14506, 0.05453, 0.007883,
    0.002967, 0.02052, 0.05453, 0.07523, 0.05453, 0.02052, 0.002967,
    0.000429, 0.002967, 0.007883, 0.01087, 0.007883, 0.002967, 0.000429
};

const double gauss7x7_4_0[] = {
    0.00365, 0.00916, 0.01582, 0.01825, 0.01582, 0.00916, 0.00365,
    0.00916, 0.02301, 0.0398, 0.04587, 0.0398, 0.02301, 0.00916,
    0.01582, 0.0398, 0.06883, 0.07918, 0.06883, 0.0398, 0.01582,
    0.01825, 0.04587, 0.07918, 0.09113, 0.07918, 0.04587, 0.01825,
    0.01582, 0.0398, 0.06883, 0.07918, 0.06883, 0.0398, 0.01582,
    0.00916, 0.02301, 0.0398, 0.04587, 0.0398, 0.02301, 0.00916,
    0.00365, 0.00916, 0.01582, 0.01825, 0.01582, 0.00916, 0.00365
};


int write_pgm(int** matrix, int width, int height, const char* filename, int is_binary) {
	int i, j;
	
	FILE* file = fopen(filename, is_binary ? "wb" : "w");
    if (!file) {
        fprintf(stderr, "Error: Could not create %s!\n", filename);
        return -1;
    }

    // Write header
    fprintf(file, "P%d\n", is_binary ? 5 : 2);
    fprintf(file, "# Created by PGM writer\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    if (is_binary) {
        // Write binary data
        unsigned char* buffer = (unsigned char*)malloc(width);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                buffer[j] = (unsigned char)matrix[i][j];
            }
            fwrite(buffer, 1, width, file);
        }
        free(buffer);
    } else {
        // Write ASCII data
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                fprintf(file, "%d ", matrix[i][j]);
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
    return 0;
}

int** read_pgm(const char* filename, int* width, int* height) {
	int i, j;
	
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open file!\n");
        return NULL;
    }

    char line[MAX_LINE];
    int is_binary = 0;
    int max_val = 255;

    // Read magic number
    if (!fgets(line, sizeof(line), file)) {
        printf("Invalid file!\n");
        fclose(file);
        return NULL;
    }

    if (strncmp(line, "P5", 2) == 0) {
        is_binary = 1;
    } else if (strncmp(line, "P2", 2) != 0) {
        printf("Invalid PGM format (must be P2 or P5)!\n");
        fclose(file);
        return NULL;
    }

    // Skip comments
    do {
        if (!fgets(line, sizeof(line), file)) {
            printf("File error!\n");
            fclose(file);
            return NULL;
        }
    } while (line[0] == '#');

    // Read width and height
    sscanf(line, "%d %d", width, height);

    // Read max value
    if (!fgets(line, sizeof(line), file)) {
        printf("File error!\n");
        fclose(file);
        return NULL;
    }
    sscanf(line, "%d", &max_val);

    // Allocate memory
    int** matrix = (int**)malloc(*height * sizeof(int*));
    for (i = 0; i < *height; i++) {
        matrix[i] = (int*)malloc(*width * sizeof(int));
    }

    if (is_binary) {
        // Read binary data
        unsigned char* buffer = (unsigned char*)malloc(*width);
        for (i = 0; i < *height; i++) {
            if (fread(buffer, 1, *width, file) != *width) {
                printf("Error reading binary data!\n");
                free(buffer);
                fclose(file);
                return NULL;
            }
            for (j = 0; j < *width; j++) {
                matrix[i][j] = buffer[j];
            }
        }
        free(buffer);
    } else {
        // Read ASCII data
        for (i = 0; i < *height; i++) {
            for (j = 0; j < *width; j++) {
                int val;
                if (fscanf(file, "%d", &val) != 1) {
                    printf("Error reading pixel data!\n");
                    fclose(file);
                    return NULL;
                }
                matrix[i][j] = val;
            }
        }
    }

    fclose(file);
    return matrix;
}

// Matrisin içeriðini yazdýran fonksiyon
void print_matrix(int** matrix, int width, int height) {
    int i, j;
     
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int** apply_filter(int** input_matrix, int width, int height, const double* filter, int filter_size) {
    int i, j, k, l;
    int margin = filter_size / 2;
    
    // Çýktý matrisi için bellek ayýrma
    int** output_matrix = (int**)malloc(height * sizeof(int*));
    for (i = 0; i < height; i++) {
        output_matrix[i] = (int*)malloc(width * sizeof(int));
    }

    // Filtreleme iþlemi
    for (i = margin; i < height - margin; i++) {
        for (j = margin; j < width - margin; j++) {
            double sum = 0.0;
            
            // Filtre penceresinde dolaþ
            for (k = -margin; k <= margin; k++) {
                for (l = -margin; l <= margin; l++) {
                    // Filtre indeksi: (k + margin) * filter_size + (l + margin)
                    sum += input_matrix[i + k][j + l] * filter[(k + margin) * filter_size + (l + margin)];
                }
            }
            
            // Deðeri [0-255] aralýðýna kýrp ve yuvarla
            sum = fmax(0, fmin(255, sum));
            output_matrix[i][j] = (int)(sum + 0.5);
        }
    }

    // Kenarlarý kopyala (filtre uygulanmayan bölgeler)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (i < margin || i >= height - margin || 
                j < margin || j >= width - margin) {
                output_matrix[i][j] = input_matrix[i][j];
            }
        }
    }

    return output_matrix;
}

int** apply_laplacian_filter(int** input_matrix, int width, int height, const double* kernel, int kernel_size) {
	int i, j, k, l;
    int margin = kernel_size / 2;
    int** output_matrix = (int**)malloc(height * sizeof(int*));
    
    // Bellek ayýrma
    for (i = 0; i < height; i++) {
        output_matrix[i] = (int*)malloc(width * sizeof(int));
    }

    // Filtreleme iþlemi
    for (i = margin; i < height - margin; i++) {
        for (j = margin; j < width - margin; j++) {
            double sum = 0.0;
            
            // Kernel penceresinde dolaþ
            for (k = -margin; k <= margin; k++) {
                for (l = -margin; l <= margin; l++) {
                    sum += input_matrix[i + k][j + l] * kernel[(k + margin) * kernel_size + (l + margin)];
                }
            }
            
            // Negatif deðerleri ve aþýrý pozitif deðerleri kýrp
            sum = fmax(0, fmin(255, fabs(sum))); // Mutlak deðer al ve [0-255] aralýðýna kýrp
            output_matrix[i][j] = (int)(sum + 0.5);
        }
    }

    // Kenarlarý orijinal deðerlerle doldur (opsiyonel)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (i < margin || i >= height - margin || j < margin || j >= width - margin) {
                output_matrix[i][j] = input_matrix[i][j];
            }
        }
    }

    return output_matrix;
}

// Ana fonksiyon
int main() {
    int width, height, i;
    
    /*
    int** pgm_matrix = read_pgm("saturn.ascii.pgm", &width, &height);
    
    if (!pgm_matrix) {
        printf("Görüntü yüklenemedi!\n");
        return -1;
    }
	
    // 1. Orijinal görüntüye Laplacian uygula
    int** laplacian1_orig = apply_laplacian_filter(pgm_matrix, width, height, laplacian_kernel1, 3);
    int** laplacian2_orig = apply_laplacian_filter(pgm_matrix, width, height, laplacian_kernel2, 3);
    write_pgm(laplacian1_orig, width, height, "laplacian1_original.pgm",0);
    write_pgm(laplacian2_orig, width, height, "laplacian2_original.pgm",0);

    // 2. Tüm Gauss filtrelenmiþ görüntülere Laplacian uygula
    // 3x3 filtreler
    int** gauss3x3_1 = apply_filter(pgm_matrix, width, height, gauss3x3_1_0, 3);
    int** laplacian1_3x3_1 = apply_laplacian_filter(gauss3x3_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_1 = apply_laplacian_filter(gauss3x3_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_1, width, height, "gauss3x3_1.pgm", 0);
    write_pgm(laplacian1_3x3_1, width, height, "laplacian1_gauss3x3_1.pgm",0);
    write_pgm(laplacian2_3x3_1, width, height, "laplacian2_gauss3x3_1.pgm",0);

    int** gauss3x3_2 = apply_filter(pgm_matrix, width, height, gauss3x3_2_0, 3);
    int** laplacian1_3x3_2 = apply_laplacian_filter(gauss3x3_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_2 = apply_laplacian_filter(gauss3x3_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_2, width, height, "gauss3x3_2.pgm",0);
    write_pgm(laplacian1_3x3_2, width, height, "laplacian1_gauss3x3_2.pgm",0);
    write_pgm(laplacian2_3x3_2, width, height, "laplacian2_gauss3x3_2.pgm",0);

    int** gauss3x3_4 = apply_filter(pgm_matrix, width, height, gauss3x3_4_0, 3);
    int** laplacian1_3x3_4 = apply_laplacian_filter(gauss3x3_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_4 = apply_laplacian_filter(gauss3x3_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_4, width, height, "gauss3x3_4.pgm",0);
    write_pgm(laplacian1_3x3_4, width, height, "laplacian1_gauss3x3_4.pgm",0);
    write_pgm(laplacian2_3x3_4, width, height, "laplacian2_gauss3x3_4.pgm",0);

    // 5x5 filtreler
    int** gauss5x5_1 = apply_filter(pgm_matrix, width, height, gauss5x5_1_0, 5);
    int** laplacian1_5x5_1 = apply_laplacian_filter(gauss5x5_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_1 = apply_laplacian_filter(gauss5x5_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_1, width, height, "gauss5x5_1.pgm",0);
    write_pgm(laplacian1_5x5_1, width, height, "laplacian1_gauss5x5_1.pgm",0);
    write_pgm(laplacian2_5x5_1, width, height, "laplacian2_gauss5x5_1.pgm",0);

    int** gauss5x5_2 = apply_filter(pgm_matrix, width, height, gauss5x5_2_0, 5);
    int** laplacian1_5x5_2 = apply_laplacian_filter(gauss5x5_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_2 = apply_laplacian_filter(gauss5x5_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_2, width, height, "gauss5x5_2.pgm",0);
    write_pgm(laplacian1_5x5_2, width, height, "laplacian1_gauss5x5_2.pgm",0);
    write_pgm(laplacian2_5x5_2, width, height, "laplacian2_gauss5x5_2.pgm",0);

    int** gauss5x5_4 = apply_filter(pgm_matrix, width, height, gauss5x5_4_0, 5);
    int** laplacian1_5x5_4 = apply_laplacian_filter(gauss5x5_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_4 = apply_laplacian_filter(gauss5x5_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_4, width, height, "gauss5x5_4.pgm",0);
    write_pgm(laplacian1_5x5_4, width, height, "laplacian1_gauss5x5_4.pgm",0);
    write_pgm(laplacian2_5x5_4, width, height, "laplacian2_gauss5x5_4.pgm",0);

    // 7x7 filtreler
    int** gauss7x7_1 = apply_filter(pgm_matrix, width, height, gauss7x7_1_0, 7);
    int** laplacian1_7x7_1 = apply_laplacian_filter(gauss7x7_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_1 = apply_laplacian_filter(gauss7x7_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_1, width, height, "gauss7x7_1.pgm",0);
    write_pgm(laplacian1_7x7_1, width, height, "laplacian1_gauss7x7_1.pgm",0);
    write_pgm(laplacian2_7x7_1, width, height, "laplacian2_gauss7x7_1.pgm",0);

    int** gauss7x7_2 = apply_filter(pgm_matrix, width, height, gauss7x7_2_0, 7);
    int** laplacian1_7x7_2 = apply_laplacian_filter(gauss7x7_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_2 = apply_laplacian_filter(gauss7x7_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_2, width, height, "gauss7x7_2.pgm",0);
    write_pgm(laplacian1_7x7_2, width, height, "laplacian1_gauss7x7_2.pgm",0);
    write_pgm(laplacian2_7x7_2, width, height, "laplacian2_gauss7x7_2.pgm",0);

    int** gauss7x7_4 = apply_filter(pgm_matrix, width, height, gauss7x7_4_0, 7);
    int** laplacian1_7x7_4 = apply_laplacian_filter(gauss7x7_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_4 = apply_laplacian_filter(gauss7x7_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_4, width, height, "gauss7x7_4.pgm",0);
    write_pgm(laplacian1_7x7_4, width, height, "laplacian1_gauss7x7_4.pgm",0);
    write_pgm(laplacian2_7x7_4, width, height, "laplacian2_gauss7x7_4.pgm",0);
	
	*/
	
	int** pgm_matrix = read_pgm("fruit.pgm", &width, &height);
    
    if (!pgm_matrix) {
        printf("Görüntü yüklenemedi!\n");
        return -1;
    }
	
    // 1. Orijinal görüntüye Laplacian uygula
    int** laplacian1_orig = apply_laplacian_filter(pgm_matrix, width, height, laplacian_kernel1, 3);
    int** laplacian2_orig = apply_laplacian_filter(pgm_matrix, width, height, laplacian_kernel2, 3);
    write_pgm(laplacian1_orig, width, height, "laplacian1_original.pgm",1);
    write_pgm(laplacian2_orig, width, height, "laplacian2_original.pgm",1);

    // 2. Tüm Gauss filtrelenmiþ görüntülere Laplacian uygula
    // 3x3 filtreler
    int** gauss3x3_1 = apply_filter(pgm_matrix, width, height, gauss3x3_1_0, 3);
    int** laplacian1_3x3_1 = apply_laplacian_filter(gauss3x3_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_1 = apply_laplacian_filter(gauss3x3_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_1, width, height, "gauss3x3_1.pgm", 1);
    write_pgm(laplacian1_3x3_1, width, height, "laplacian1_gauss3x3_1.pgm",1);
    write_pgm(laplacian2_3x3_1, width, height, "laplacian2_gauss3x3_1.pgm",1);

    int** gauss3x3_2 = apply_filter(pgm_matrix, width, height, gauss3x3_2_0, 3);
    int** laplacian1_3x3_2 = apply_laplacian_filter(gauss3x3_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_2 = apply_laplacian_filter(gauss3x3_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_2, width, height, "gauss3x3_2.pgm",1);
    write_pgm(laplacian1_3x3_2, width, height, "laplacian1_gauss3x3_2.pgm",1);
    write_pgm(laplacian2_3x3_2, width, height, "laplacian2_gauss3x3_2.pgm",1);

    int** gauss3x3_4 = apply_filter(pgm_matrix, width, height, gauss3x3_4_0, 3);
    int** laplacian1_3x3_4 = apply_laplacian_filter(gauss3x3_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_3x3_4 = apply_laplacian_filter(gauss3x3_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss3x3_4, width, height, "gauss3x3_4.pgm",1);
    write_pgm(laplacian1_3x3_4, width, height, "laplacian1_gauss3x3_4.pgm",1);
    write_pgm(laplacian2_3x3_4, width, height, "laplacian2_gauss3x3_4.pgm",1);

    // 5x5 filtreler
    int** gauss5x5_1 = apply_filter(pgm_matrix, width, height, gauss5x5_1_0, 5);
    int** laplacian1_5x5_1 = apply_laplacian_filter(gauss5x5_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_1 = apply_laplacian_filter(gauss5x5_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_1, width, height, "gauss5x5_1.pgm",1);
    write_pgm(laplacian1_5x5_1, width, height, "laplacian1_gauss5x5_1.pgm",1);
    write_pgm(laplacian2_5x5_1, width, height, "laplacian2_gauss5x5_1.pgm",1);

    int** gauss5x5_2 = apply_filter(pgm_matrix, width, height, gauss5x5_2_0, 5);
    int** laplacian1_5x5_2 = apply_laplacian_filter(gauss5x5_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_2 = apply_laplacian_filter(gauss5x5_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_2, width, height, "gauss5x5_2.pgm",1);
    write_pgm(laplacian1_5x5_2, width, height, "laplacian1_gauss5x5_2.pgm",1);
    write_pgm(laplacian2_5x5_2, width, height, "laplacian2_gauss5x5_2.pgm",1);

    int** gauss5x5_4 = apply_filter(pgm_matrix, width, height, gauss5x5_4_0, 5);
    int** laplacian1_5x5_4 = apply_laplacian_filter(gauss5x5_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_5x5_4 = apply_laplacian_filter(gauss5x5_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss5x5_4, width, height, "gauss5x5_4.pgm",1);
    write_pgm(laplacian1_5x5_4, width, height, "laplacian1_gauss5x5_4.pgm",1);
    write_pgm(laplacian2_5x5_4, width, height, "laplacian2_gauss5x5_4.pgm",1);

    // 7x7 filtreler
    int** gauss7x7_1 = apply_filter(pgm_matrix, width, height, gauss7x7_1_0, 7);
    int** laplacian1_7x7_1 = apply_laplacian_filter(gauss7x7_1, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_1 = apply_laplacian_filter(gauss7x7_1, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_1, width, height, "gauss7x7_1.pgm",1);
    write_pgm(laplacian1_7x7_1, width, height, "laplacian1_gauss7x7_1.pgm",1);
    write_pgm(laplacian2_7x7_1, width, height, "laplacian2_gauss7x7_1.pgm",1);

    int** gauss7x7_2 = apply_filter(pgm_matrix, width, height, gauss7x7_2_0, 7);
    int** laplacian1_7x7_2 = apply_laplacian_filter(gauss7x7_2, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_2 = apply_laplacian_filter(gauss7x7_2, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_2, width, height, "gauss7x7_2.pgm",1);
    write_pgm(laplacian1_7x7_2, width, height, "laplacian1_gauss7x7_2.pgm",1);
    write_pgm(laplacian2_7x7_2, width, height, "laplacian2_gauss7x7_2.pgm",1);

    int** gauss7x7_4 = apply_filter(pgm_matrix, width, height, gauss7x7_4_0, 7);
    int** laplacian1_7x7_4 = apply_laplacian_filter(gauss7x7_4, width, height, laplacian_kernel1, 3);
    int** laplacian2_7x7_4 = apply_laplacian_filter(gauss7x7_4, width, height, laplacian_kernel2, 3);
    write_pgm(gauss7x7_4, width, height, "gauss7x7_4.pgm",1);
    write_pgm(laplacian1_7x7_4, width, height, "laplacian1_gauss7x7_4.pgm",1);
    write_pgm(laplacian2_7x7_4, width, height, "laplacian2_gauss7x7_4.pgm",1);
	




    // Belleði serbest býrak
    for (i = 0; i < height; i++) {
        free(pgm_matrix[i]);
        free(laplacian1_orig[i]); 
        free(laplacian2_orig[i]);
        free(gauss3x3_1[i]); 
        free(laplacian1_3x3_1[i]); 
        free(laplacian2_3x3_1[i]);
        free(gauss3x3_2[i]); 
        free(laplacian1_3x3_2[i]); 
        free(laplacian2_3x3_2[i]);
        free(gauss3x3_4[i]); 
        free(laplacian1_3x3_4[i]); 
        free(laplacian2_3x3_4[i]);
        free(gauss5x5_1[i]); 
        free(laplacian1_5x5_1[i]); 
        free(laplacian2_5x5_1[i]);
        free(gauss5x5_2[i]); 
        free(laplacian1_5x5_2[i]); 
        free(laplacian2_5x5_2[i]);
        free(gauss5x5_4[i]); 
        free(laplacian1_5x5_4[i]); 
        free(laplacian2_5x5_4[i]);
        free(gauss7x7_1[i]); 
        free(laplacian1_7x7_1[i]); 
        free(laplacian2_7x7_1[i]);
        free(gauss7x7_2[i]); 
        free(laplacian1_7x7_2[i]); 
        free(laplacian2_7x7_2[i]);
        free(gauss7x7_4[i]); 
        free(laplacian1_7x7_4[i]); 
        free(laplacian2_7x7_4[i]);
    }
    
    free(pgm_matrix);
    free(laplacian1_orig); 
    free(laplacian2_orig);
    free(gauss3x3_1); 
    free(laplacian1_3x3_1); 
    free(laplacian2_3x3_1);
    free(gauss3x3_2); 
    free(laplacian1_3x3_2); 
    free(laplacian2_3x3_2);
    free(gauss3x3_4); 
    free(laplacian1_3x3_4); 
    free(laplacian2_3x3_4);
    free(gauss5x5_1); 
    free(laplacian1_5x5_1); 
    free(laplacian2_5x5_1);
    free(gauss5x5_2); 
    free(laplacian1_5x5_2); 
    free(laplacian2_5x5_2);
    free(gauss5x5_4); 
    free(laplacian1_5x5_4); 
    free(laplacian2_5x5_4);
    free(gauss7x7_1); 
    free(laplacian1_7x7_1); 
    free(laplacian2_7x7_1);
    free(gauss7x7_2); 
    free(laplacian1_7x7_2); 
    free(laplacian2_7x7_2);
    free(gauss7x7_4); 
    free(laplacian1_7x7_4); 
    free(laplacian2_7x7_4);
    return 0;
}
