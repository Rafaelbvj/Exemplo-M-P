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

//ReLU as an activation function
#define FUNC_ATIVA_RLU(a)       ((a>=0)?a:(a*0.01f))
#define FUNC_ATIVA_RLU_DER(a)   ((a>=0)?1:0.01f)
#define FUNC_ATIVA_RLU_DER_F(a) ((a>=0)?1:0.01f)

//Tangente hiperb�lico
#define FUNC_ATIVA_TANH(a)           tanh(a)
#define FUNC_ATIVA_TANH_DER(a)       1-pow(tanh(a),2)
#define FUNC_ATIVA_TANH_DER_F(a)     1-pow(a,2)


typedef long double TYPE;
typedef TYPE(*Func)(TYPE);

typedef struct act_func{
    Func activation_func;           //funcao de ativacao
    Func activation_func_der;       //derivada
    Func activation_func_der_f;     //derivada com o valor de retorno da funcao de ativacao como parametro
}ActivationFunction;

class PE
{
public:

    std::vector <PE*> pe;          //elementos de processos conectados
    std::vector <TYPE> we ;        //pesos
    std::vector <TYPE> we_old;     //pesos antes do ultimo ajuste
    TYPE my_value;                 //valor elemento de processo, somatorio (Wi*Xi)
    long double my_value_f;        //valor elemento de processo na funcao de ativacao
    TYPE bi ;                      //bias coeficiente linear

    // ef*fu*uw = erro instantaneo ou gradiente (dJ/dw)
    TYPE ef;                       //Derivada do erro total em relacao a fun�ao separadora/ativacao
    TYPE fu;                       //Derivada da funcao separadora em relacao ao valor do elemento de processo
    TYPE uw;                       //Derivada do elemento de processo em relacao ao peso
    ActivationFunction *s_func;

    PE():my_value{0},my_value_f{0},s_func{nullptr}
    {
        srand(time(NULL));
    }

    //Apenas para a camada de entrada
    void set_activation_function(ActivationFunction *m){
        s_func = m;
    }

    TYPE calc_value(TYPE b)
    {
        my_value = 0;
        bi = b;
        size_t size_pe = pe.size();
        for(size_t i=0; i<size_pe; i++)
        {
            my_value+=pe[i]->my_value_f * we[i] ;
        }
        my_value += bi;
        my_value_f = s_func->activation_func(my_value);
        return my_value_f;
    }

    TYPE calc_value_SIGMOID(TYPE b)
    {
        my_value = 0;
        bi = b;
        size_t size_pe = pe.size();
        for(size_t i=0; i<size_pe; i++)
        {
            my_value+=pe[i]->my_value_f * we[i] ;
        }
        my_value += bi;

        my_value_f = FUNC_ATIVA_LOG(my_value);
        return my_value_f;
    }
    TYPE calc_value_ReLU(TYPE b)
    {
        my_value = 0;
        bi = b;
        size_t size_pe = pe.size();
        for(size_t i=0; i<size_pe; i++)
        {
            my_value+=pe[i]->my_value_f * we[i] ;

        }
        my_value += bi;

        my_value_f = FUNC_ATIVA_RLU(my_value);
        return my_value_f;
    }

    float generate_weight(){
        return (rand()%100)/100.0f;
    }
    bool compare_output(int index_weight,TYPE output, TYPE step_size)
    {
        
        ef = -(output - my_value_f);
        if(ef == 0)
        {
            //Resposta desejada, nao � necessario ajustes
            return true;
        }
        fu = s_func->activation_func_der_f(my_value_f);

        uw = pe[index_weight]->my_value_f;

        we_old[index_weight] = we[index_weight];

        we[index_weight] -= step_size*ef*fu*uw;

        return false;

    }

    bool compare_output_SIGMOID(int index_weight,TYPE output, TYPE step_size)
    {
        ef = -(output - my_value_f);
        if(ef == 0)
        {
            //Resposta desejada, nao � necessario ajustes
            return true;
        }
        fu = FUNC_ATIVA_LOG_DER_F(my_value_f);

        uw = pe[index_weight]->my_value_f;

        we_old[index_weight] = we[index_weight];

        we[index_weight] -= step_size*ef*fu*uw;

        return false;
    }
    bool compare_output_ReLU(int index_weight,TYPE output, TYPE step_size)
    {
        ef = (my_value_f - output);        //Cost function - derivada do erro quadratico medio
        if(ef == 0)
        {
            //Resposta desejada, nao � necessario ajustes
            return true;
        }

        fu = FUNC_ATIVA_RLU_DER_F(my_value_f);

        uw = pe[index_weight]->my_value_f;

        we_old[index_weight] = we[index_weight];

        we[index_weight] -= step_size*ef*fu*uw;

        return false;
    }


    bool connect_pe(PE *p, TYPE weight)
    {
        if(p == NULL)
        {
            return false;
        }
        
        s_func = p->s_func;
        
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
TYPE largest_step_size(std::vector <PE*> input)
{
    TYPE lss =0;
    size_t in_size = input.size();
    for(size_t i=0; i<in_size; i++)
    {
        lss+=pow(input[i]->my_value_f, 2);
    }
    lss =(2*input.size())/lss;
    return lss;
}
TYPE fastest_step_size(std::vector <PE*> input)
{
    return largest_step_size(input)/2.0f;

}


bool compare_forward (PE& first, std::vector <PE> previous_layer, int index_weight, TYPE step_size)
{

    std::vector <PE>::iterator it = previous_layer.begin();
    first.ef  = 0;
    for(size_t c; it!=previous_layer.end(); it++)
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
            //Elemento nao est� conectado � camada
            return false;
        }
        first.ef += (*it).ef  *  (*it).we_old[c] * first.s_func->activation_func_der_f((*it).my_value_f);

    }

    first.fu = first.s_func->activation_func_der_f(first.my_value_f);

    first.uw = first.pe[index_weight]->my_value_f;

    first.we_old[index_weight] = first.we[index_weight];

    first.we[index_weight] -= first.ef*first.fu*first.uw*step_size;

    return true;
}

bool compare_forward_ReLU(PE& first, std::vector <PE> previous_layer, int index_weight, TYPE step_size)
{

    std::vector <PE>::iterator it = previous_layer.begin();
    first.ef  = 0;
    for(size_t c; it!=previous_layer.end(); it++)
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
            //Elemento nao est� conectado � camada
            return false;
        }
        first.ef += (*it).ef  *  (*it).we_old[c] * FUNC_ATIVA_RLU_DER_F((*it).my_value_f);
    }

    first.fu = FUNC_ATIVA_RLU_DER_F(first.my_value_f);

    first.uw = first.pe[index_weight]->my_value_f;

    first.we_old[index_weight] = first.we[index_weight];

    first.we[index_weight] -= first.ef*first.fu*first.uw*step_size;

    return true;
}

bool compare_forward_SIGMOID(PE& first, std::vector <PE> previous_layer, int index_weight, TYPE step_size)
{

    std::vector <PE>::iterator it = previous_layer.begin();
    first.ef  = 0;
    for(size_t c; it!=previous_layer.end(); it++)
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
            //Elemento nao est� conectado � camada
            return false;
        }
        first.ef += (*it).ef  *  (*it).we_old[c]  *  FUNC_ATIVA_LOG_DER_F((*it).my_value_f);
    }


    first.fu = FUNC_ATIVA_LOG_DER_F(first.my_value_f);

    first.uw = first.pe[index_weight]->my_value_f;

    first.we_old[index_weight] = first.we[index_weight];

    first.we[index_weight] -= first.ef*first.fu*first.uw*step_size;

    return true;
}
#endif // PERCEPTRON_H
