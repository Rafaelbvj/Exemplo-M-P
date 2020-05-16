#ifndef IMAGE_H
#define IMAGE_H
#include "perceptron.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
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
    PE output;
    PE *input;

    long double result_calc;
    map <long double,int> id_t;
    vector <long double> dif_obtido;

    void pixel_aloc(Pixel ***p, unsigned int col, unsigned int row)
    {
        *p = (Pixel**)malloc(sizeof(Pixel*)*col);
        for(unsigned int i=0; i<col; i++)
        {
            (*p)[i] = (Pixel*)malloc(sizeof(Pixel)*row);
        }
    }
    void pixel_free(Pixel ***p, int row)
    {
        for(unsigned int i=0; i<row; i++)
        {
            free( (*p)[i]);
        }
        free(*p);

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
        filepath = path;
        FILE *file = fopen(path,"rb");
        if(file == NULL)
        {
            cout<<"Erro ao abrir o arquivo!"<<endl;
            return errno;
        }
        if(pixels !=NULL)
        {
            pixel_free(&pixels,b.biHeight);
        }
        char id[3] = {0,0,0};
        fread(id, sizeof(char),2, file);
        fread(&b, sizeof(M_BITMAP),1, file);
        if(strcmp(id,"BM") !=0 )
        {
            cout<<"Arquivo não é BITMAP!"<<endl;
            return -1;
        }
        cout<<"File:"<<path<<endl;
        cout<<"File size:"<<b.bfSize<<endl;
        cout<<"Image height:"<<b.biHeight<<endl;
        cout<<"Image width:"<<b.biWidth<<endl;

        pixel_aloc(&pixels, b.biHeight, b.biWidth);
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

    int identify(int id)
    {
        const int n_elements = 5;
        for(unsigned int u=0; u<n_elements; u++)
        {
            for(unsigned int i=0,t =0; i<b.biHeight; i++)
            {
                for(unsigned int j=0; j<b.biWidth; j++, t++)
                {
                    if(pixels[i][j].B==1){
                        input[u].pe[t]->my_value_f = j;
                    }
                    else{
                        input[u].pe[t]->my_value_f = 1;
                    }
                }

            }
            input[u].calc_value(0);

        }

        output.calc_value(0);
        dif_obtido.push_back(fabs(output.my_value_f-result_calc));
        id_t.insert(make_pair(*(dif_obtido.end()-1),id));

        cout<<"Identificado:"<<output.my_value_f<<endl;
    }

    int give_min(){
        sort(dif_obtido.begin(),dif_obtido.end());
        return id_t[dif_obtido[0]];
    }

    int create_layer()
    {
        const int n_elements = 5;          //Número de elementos na "hidden layer"
        unsigned int sz = b.biHeight*b.biWidth*n_elements;
        unsigned int szwh = b.biHeight*b.biWidth;
        input = new PE[szwh];
        PE *setpe = new PE[sz];
        std::vector <PE> hidden_layer;
        for(unsigned int u=0,k=0; u<n_elements; u++)
        {
            for(unsigned int i=0; i<b.biHeight; i++)
            {
                for(unsigned int j=0; j<b.biWidth; j++, k++)
                {
                    //Criterio de entrada (Pixel == 1 "escuro" ) -> armazena X (width)
                    if(pixels[i][j].B==1){
                        setpe[k].my_value_f = j;
                    }
                    else{
                        setpe[k].my_value_f = 1;
                    }
                    input[u].connect_pe(&setpe[k],input[u].generate_weight()/10000.0f); //Peso diminuido para evitar saturação da funcao de ativaçao
                }

            }
            input[u].calc_value(0);
        }
        long double result = 0.7;               //Resultado esperado sempre deve ser <1, por causa da funcao sigmoid.

        long double step = fastest_step_size(input[0].pe);
        cout<<"Step size:"<<step<<endl;
        for(unsigned int i=0; i<n_elements; i++)
        {
            output.connect_pe(&input[i], output.generate_weight());
        }

        vector <PE> outputs;

        output.calc_value(0);
        for(int epoc =0 ; epoc<50; epoc++)
        {

            for(unsigned int i=0; i<n_elements; i++)
            {
                output.compare_output(i, result, step);
            }

            outputs.push_back(output);

            for(unsigned int i=0; i<n_elements; i++)
            {

                for(int e =0 ; e<szwh; e++)
                {
                    compare_forward(input[i],outputs,e,step);
                }
            }

            for(unsigned int e=0; e<n_elements; e++)
            {
                input[e].calc_value(0);
            }

            outputs.clear();

            output.calc_value(0);
            cout<<"Epoca:"<<epoc+1<<endl;
            cout<<output.my_value_f<<endl;
        }
        result_calc = output.my_value_f;


    }
    //only two colors
    int set_binary_pixel()
    {
        if(pixels == NULL)
        {
            return 0;
        }

        for(unsigned int i=0; i<b.biHeight; i++)
        {
            for(unsigned int j=0; j<b.biWidth; j++)
            {
                if(pixels[i][j].B != 0xFF ||pixels[i][j].G != 0xFF||pixels[i][j].R != 0xFF)
                {
                    pixels[i][j].B  = 1;
                    pixels[i][j].G  = 1;
                    pixels[i][j].R  = 1;
                }
            }
        }
        int npad;
        for(npad =0; ((b.biWidth)*3+npad) % 4 !=0 ; npad++) {}
        char *pad = new char[npad];
        FILE *file = fopen("teste2.bmp","wb");  //replace filepath
        char id[3] = {'B','M'};
        fwrite(id,1,2,file);
        fwrite(&b,sizeof(M_BITMAP),1,file);
        for(unsigned int i=0; i<b.biHeight; i++)
        {
            fwrite(pixels[i],sizeof(Pixel), b.biWidth,file);
            fwrite(pad,npad, 1,file);
        }
        fclose(file);
        delete []pad;

    }

    ~Image()
    {
        pixel_free(&pixels,b.biHeight);
    }

};
#endif // IMAGE_H
