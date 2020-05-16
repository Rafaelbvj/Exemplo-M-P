#include <iostream>
#include <stdio.h>
#include <string.h>

#include "image.hpp"
#include "perceptron.hpp"
#include <windows.h>

using namespace std;

int main()
{
    HANDLE hProc = GetCurrentProcess();
    if(!SetPriorityClass(hProc,REALTIME_PRIORITY_CLASS)){
        cout<<"Erro prioridade!"<<endl;
        return -1;
    }
    //Imagem de analise
    Image im("teste.bmp");
    im.set_binary_pixel();
    im.create_layer();

    //Imagens para comparação
    im.open_file("ver0.bmp");
    im.set_binary_pixel();
    im.identify(0);     //numero identificador
    cout<<endl<<endl<<endl;

    im.open_file("ver1.bmp");
    im.set_binary_pixel();
    im.identify(1);     //numero identificador
    cout<<endl<<endl<<endl;

    im.open_file("ver2.bmp");
    im.set_binary_pixel();
    im.identify(2);     //numero identificador
    cout<<endl<<endl<<endl;


    cout<<"Resultado mais proximo:"<<im.give_min()<<endl;

    return 0;
}

