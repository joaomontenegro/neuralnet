#include "NeuralNet.h"
#include "Tests.h"
#include "TrainIncrement.h"
#include "Mnist.h"

#include <iostream>
#include <string>



int main(int argc, char* argv[])
{
	//trainIncrement(argc, argv);

	MNIST mnist;
	NeuralNet* net = mnist.train(argv[1], argv[2], 0.5, 0.5, 60000, 20);

	for (size_t l = 0 ; l < mnist.images.size(); ++l)
	{
		//mnist.printImage(l, 500);
		std::cout << "EXPECTED: " << l << "   DETECTED: " << mnist.detect(mnist.images[l][500]) << std::endl << std::endl;
	}
	

	return 0;
}
