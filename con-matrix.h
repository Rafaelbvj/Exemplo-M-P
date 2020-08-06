#ifndef CON_MATRIX_H_INCLUDED
#define CON_MATRIX_H_INCLUDED
#include "image.hpp"
#include <math.h>

enum color_opt
{
    RED_COLOR,
    BLUE_COLOR,
    GREEN_COLOR
};


template <typename Data>
class Matrix
{
private:
    size_t width;
    size_t height;
    Data **data;

    void **matrix_alloc(size_t sizef, size_t col, size_t row)
    {
        void **p;
        p = (void**)calloc(row,sizeof(void*));
        for(size_t i=0; i<row; i++)
        {
            p[i] = calloc(col,sizef);
        }
        return (void**)p;
    }

    void matrix_free(void **p, size_t row)
    {
        for(size_t i=0; i<row; i++)
        {
            free( p[i]);
        }
        free(p);

    }
    bool isvalid(size_t x,size_t y)
    {
        if(x>=height || y>=width || x < 0 || y < 0)
        {
            //out of range
            return false;
        }
        return true;
    }
public:
    Matrix(size_t w = 4, size_t h = 4):width{w}, height{h}
    {
        data = (Data**)matrix_alloc(sizeof(Data),width,height);
        srand(time(NULL));

    }
    ~Matrix()
    {
        matrix_free((void**)data,height);

    }
    void free_data()
    {

        matrix_free((void**)data,height);

    }

    bool set_value_XY(size_t x,size_t y, Data d)
    {
        if(!isvalid(x,y))
        {
            return false;
        }
        data[x][y] = d;
        return true;
    }
    Data get_value_XY(size_t x,size_t y) const
    {
        return isvalid(x,y)?data[x][y]:0;
    }
    size_t get_height() const
    {
        return height;
    }
    size_t get_width() const
    {
        return width;
    }
    Data **get_data() const
    {
        return data;
    }

    bool select_color(Image &im, int opt)
    {
        if(im.pixels == NULL)
        {
            return 0;
        }
        if(data != NULL)
        {
            matrix_free((void**)data,height);
        }
        height = im.b.biHeight;
        width = im.b.biWidth;
        data = (Data**)matrix_alloc(sizeof(Data),width,height);
        for(size_t h =0; h<height; h++)
        {
            for(size_t w =0; w<width; w++)
            {
                switch(opt)
                {
                case RED_COLOR:
                    data[h][w] = im.pixels[h][w].R ;
                    break;
                case BLUE_COLOR:
                    data[h][w] = im.pixels[h][w].B;
                    break;
                case GREEN_COLOR:
                    data[h][w] = im.pixels[h][w].G;
                    break;

                }

            }

        }
    }
    Data* operator[] (int c)
    {
        //Safer access? isvalid() option
        return data[c];
    }

    Data sum_all_elements()
    {
        Data result =0;
        for(size_t i=0; i<height; i++)
        {
            for(size_t j=0; j<width; j++)
            {
                result += data[i][j];
            }
        }
        return result;
    }

    void set_size(size_t w,size_t h)
    {
        Matrix *result;
        result = select_region(w,0,0,h);
        matrix_free((void**)data,height);
        data = result->data;
        height = h;
        width = w;

    }

    Matrix *select_region(size_t right, size_t left, size_t top, size_t bottom)
    {
        
        size_t w ;//= abs(right-left);
        size_t h ;//= abs(bottom-top);
        Matrix *result = new Matrix(w, h);
        if(bottom > height)
        {
            bottom = height;
        }
        if(right > width)
        {
            right = width;
        }
        for(size_t i = top,c =0; i<bottom; i++,c++)
        {
            memcpy(result->data[c],&data[i][left],(right-left)*sizeof(Data));
        }

        return result;

    }


    //Multiplica�ao usual de matrizes
    Matrix *operator * (Matrix &b)
    {
        if(width != b.height)
        {
            //error
            return 0;
        }
        Matrix *result = new Matrix(b.width,height);
        for(size_t k =0; k<height; k++)
        {
            for(size_t e =0; e<b.width; e++)
            {
                for(size_t i=0; i<b.height; i++)
                {
                    (*result)[k][e] += data[k][i] * b[i][e];

                }
            }
        }
        return result;
    }


    //Opera��o de multiplicacao e soma de elemento por elemento --> aplica filtro
    Matrix *operator << (Matrix &filter)
    {
        if(filter.height!=height||filter.width!=width)
        {

            //erro
            return 0;
        }
        Matrix *result = new Matrix(width,height);

        for(size_t i=0; i<height; i++)
        {
            for(size_t j=0; j<width; j++)
            {
                (*result)[i][j] = data[i][j] * filter[i][j];
            }

        }

        return result;
    }


    //Debuging...:
    friend ostream& operator << (ostream &out, Matrix &c)
    {
        for(size_t h =0; h<c.height; h++)
        {
            for(size_t w =0; w<c.width; w++)
            {
                out<<c[h][w]<<" ";

            }
            out<<endl;
        }

        return out;

    }


};





#endif // CON-MATRIX_H_INCLUDED
