#ifndef  PERCEPTRON_H
#define PERCEPTRON_H

#include <time.h>
#include <vector>
#include <math.h>
#include <iostream>
using namespace std;

//Sigmoid as an activation function
#define FUNC_ATIVA_LOG(a) 1/(1+exp(-a))
#define FUNC_ATIVA_LOG_DER(a) FUNC_ATIVA_LOG(a)*(1-FUNC_ATIVA_LOG(a))
#define FUNC_ATIVA_LOG_DER_F(a) a*(1-a)

//RLU as an activation function
#define FUNC_ATIVA_RLU(a)       ((a>0)?a:0)
#define FUNC_ATIVA_RLU_DER(a)   ((a>0)?1:0)
#define FUNC_ATIVA_RLU_DER_F(a) ((a!=0)?1:0)

//Tangente hiperbólico
#define FUNC_ATIVA_TANH(a)           tanh(a)
#define FUNC_ATIVA_TANH_DER(a)       1-pow(tanh(a),2)
#define FUNC_ATIVA_TANH_DER_F(a)     1-pow(a,2)
class PE
{
public:

    std::vector <PE*> pe;                 //elementos de processos conectados
    std::vector <long double> we ;        //pesos
    std::vector <long double> we_old;     //pesos antes do ultimo ajuste
    long double my_value;                 //valor elemento de processo, somatorio (Wi*Xi)
    long double my_value_f;               //valor elemento de processo na funcao de ativacao
    long double bi ;                      //bias coeficiente linear

    // ef*fu*uw = erro instantaneo ou gradiente (dJ/dw)
    long double ef;                       //Derivada do erro total em relacao a funçao separadora/ativacao
    long double fu;                       //Derivada da funcao separadora em relacao ao valor do elemento de processo
    long double uw;                       //Derivada do elemento de processo em relacao ao peso

    PE():my_value{0},my_value_f{0}
    {
        srand(time(NULL));
    }

    long double calc_value(long double b)
    {
        my_value = 0;
        bi = b;
        for(unsigned int i=0; i<pe.size(); i++)
        {
            my_value+=pe[i]->my_value_f * we[i] ;

        }
        my_value += bi;
        my_value_f = FUNC_ATIVA_LOG(my_value);
        return my_value_f;
    }

    float generate_weight(){
        return (rand()%100)/float(100);
    }


    bool compare_output(int index_weight,long double output, long double step_size)
    {
        ef = -(output - my_value_f);
        if(ef == 0)
        {
            //Resposta desejada, nao é necessario ajustes
            return true;
        }
        fu = FUNC_ATIVA_LOG_DER_F(my_value_f);

        uw = pe[index_weight]->my_value_f;

        we_old[index_weight] = we[index_weight];

        we[index_weight] -= step_size*ef*fu*uw;

        return false;
    }
    bool connect_pe(PE *p, long double weight)
    {
        if(p == NULL)
        {
            return false;
        }
        pe.push_back(p);
        we.push_back(weight);
        we_old.push_back(weight);
        return true;
    }

    ~PE()
    {
        pe.clear();
        we.clear();
    }
};
long double largest_step_size(vector <PE*> input)
{
    long double lss =0;
    for(unsigned int i=0; i<input.size(); i++)
    {
        lss+=pow(input[i]->my_value_f, 2);
    }
    lss =(2*input.size())/lss;
    return lss;
}
long double fastest_step_size(vector <PE*> input)
{
    return largest_step_size(input)/2.0f;
}

bool compare_forward (PE& first, std::vector <PE> second_layer, int index_weight, long double step_size)
{

    std::vector <PE>::iterator it = second_layer.begin();
    first.ef  = 0;
    for(unsigned int c; it!=second_layer.end(); it++)
    {
        for(c=0; c<(*it).pe.size(); c++)
        {
            if((*it).pe[c] == &first)
            {
                break;
            }

        }
        if(c==(*it).pe.size())
        {
            //Elemento nao está conectado à camada
            return false;
        }
        first.ef += (*it).ef  *  (*it).we_old[c] * FUNC_ATIVA_LOG_DER_F((*it).my_value_f);
    }


    first.fu = FUNC_ATIVA_LOG_DER_F(first.my_value_f);

    first.uw = first.pe[index_weight]->my_value_f;

    first.we_old[index_weight] = first.we[index_weight];

    first.we[index_weight] -= first.ef*first.fu*first.uw*step_size;

    return true;
}
#endif // PERCEPTRON_H
