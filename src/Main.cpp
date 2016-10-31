#include "NeuralNet.h"
#include "Tests.h"
#include "TrainIncrement.h"
#include "Mnist.h"

#include <iostream>
#include <string>

void usage(char* argv0)
{
	std::cerr << " Usage: " << argv0 << " -s SAVE_FILE TRAIN_LABELS TRAIN_IMS [TEST_LABELS] [TEST_IMGS]" << std::endl << std::endl;
	std::cerr << " Usage: " << argv0 << " -l LOAD_FILE TEST_LABELS TEST_IMGS" << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	//trainIncrement(argc, argv);

	MNIST mnist;

	if (argc == 1 || argv[1][0] != '-')
	{
		usage(argv[0]);
	}

	char type = argv[1][1];
	char* testLabelsPath = 0x0;
	char* testImagesPath = 0x0;


	if (type == 's' && argc > 4)
	{

		mnist.train(argv[3], argv[4], 0.5, 0.5, 10000, 1);
		std::cout << "Saving to: " << argv[2] << std::endl;
		mnist.net->save(argv[2]);

		if (argc > 6)
		{
			testLabelsPath = argv[5];
			testImagesPath = argv[6];
		}
	}
	else if(type == 'l' && argc > 4)
	{
		mnist.load(argv[2]);
		testLabelsPath = argv[3];
		testImagesPath = argv[4];
	}
	else
	{
		std::cerr << "Error...." << std::endl;
		usage(argv[0]);
		return 0;
	}

	if (testLabelsPath != 0x0 && testImagesPath != 0x0)
	{
		std::cout << std::endl << "ERROR: " 
			<< 100 * mnist.runTestSet(testLabelsPath, testImagesPath) << "%"
			<< std::endl;;
	}

	return 0;
}
