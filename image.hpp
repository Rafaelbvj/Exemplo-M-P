#ifndef IMAGE_H
#define IMAGE_H
#include "perceptron.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

#define FILE_INFO


using namespace std;

typedef struct bmp
{
    unsigned int	bfSize;
    unsigned short	bfReserved1;
    unsigned short	bfReserved2;
    unsigned int	bfOffBits;
    unsigned int	biSize;
    long	biWidth;
    long	biHeight;
    unsigned short	biPlanes;
    unsigned short	biBitCount;
    unsigned int	biCompression;
    unsigned int	biSizeImage;
    long	biXPelsPerMeter;
    long	biYPelsPerMeter;
    unsigned int	biClrUsed;
    unsigned int	biClrImportant;
} M_BITMAP;
typedef struct pixel
{
    unsigned char B;
    unsigned char G;
    unsigned char R;
} Pixel;

class Image
{
private:
    M_BITMAP b;
    Pixel **pixels;
    char *filepath;


    void **pixel_alloc(unsigned int sizef, unsigned int col, unsigned int row)
    {
        void **p;
        p = (void**)malloc(sizeof(void*)*col);
        for(unsigned int i=0; i<col; i++)
        {
            p[i] = malloc(sizef*row);
        }
        return (void**)p;
    }
    void pixel_free(void **p, int row)
    {
        for(unsigned int i=0; i<row; i++)
        {
            free( p[i]);
        }
        free(p);

    }
public:
    Image(char *path):pixels{NULL}
    {
        open_file(path);
    }
    Image():pixels{NULL}
    {

    }
    int open_file(char *path)
    {
        FILE *file = fopen(path,"rb");
        if(file == NULL)
        {
            cerr<<"Erro ao abrir o arquivo!"<<endl;
            return errno;
        }
        filepath = path;
        if(pixels !=NULL)
        {
            pixel_free((void**)pixels,b.biHeight);
        }

        char id[3] = {0,0,0};
        fread(id, sizeof(char),2, file);
        fread(&b, sizeof(M_BITMAP),1, file);
        if( strcmp(id,"BM") !=0 )
        {
            cerr<<"Arquivo n�o � BITMAP!"<<endl;
            return -1;
        }
        #ifdef FILE_INFO
        cout<<"File:"<<path<<endl;
        cout<<"File size:"<<b.bfSize<<endl;
        cout<<"Image height:"<<b.biHeight<<endl;
        cout<<"Image width:"<<b.biWidth<<endl;
        #endif
        pixels = (Pixel**) pixel_alloc(sizeof(Pixel),b.biHeight, b.biWidth);
        int npad;
        for(npad =0; ((b.biWidth)*3+npad) % 4 !=0 ; npad++) {}
        char *pad = new char[npad];
        for(unsigned int i=0; i<b.biHeight; i++)
        {
            fread(pixels[i],sizeof(Pixel), b.biWidth,file);
            fread(pad,npad, 1,file);
        }
        fclose(file);
        delete []pad;

    }

    ~Image()
    {
        pixel_free((void**)pixels,b.biHeight);
    }

     template <typename> friend class Matrix;
};
#endif // IMAGE_H
