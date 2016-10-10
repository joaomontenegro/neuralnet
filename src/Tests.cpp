
#include "NeuralNet.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>

#define EPSILON 0.0000000001
#define DOUBLE_EQUALS(d0, d1) fabs(d0 - d1) < EPSILON

//#define VERBOSE

int RunForwardPropagateTests()
{
	Array<size_t> layerSizes(4);
	layerSizes[0] = 2;
	layerSizes[1] = 3;
	layerSizes[2] = 2;
	layerSizes[3] = 2;

	NeuralNet net(layerSizes);

	Array<NeuralNet::Sinapse> sinapses;
	sinapses.allocate(16);
	sinapses[0].weight = 0.1;
	sinapses[1].weight = 0.2;
	sinapses[2].weight = 0.3;
	sinapses[3].weight = 0.4;
	sinapses[4].weight = 0.5;
	sinapses[5].weight = 0.6;
	sinapses[6].weight = 0.7;
	sinapses[7].weight = 0.8;
	sinapses[8].weight = 0.9;
	sinapses[9].weight = 1.0;
	sinapses[10].weight = 1.1;
	sinapses[11].weight = 1.2;
	sinapses[12].weight = 1.3;
	sinapses[13].weight = 1.4;
	sinapses[14].weight = 1.5;
	sinapses[15].weight = 1.6;

	Array<NeuralNet::Neuron> neurons;
	neurons.allocate(9);
	neurons[0].bias = 0;
	neurons[1].bias = 0;
	neurons[2].bias = 0;
	neurons[3].bias = 0;
	neurons[4].bias = 0;
	neurons[5].bias = 0;
	neurons[6].bias = 0;
	neurons[7].bias = 0;
	neurons[8].bias = 0;

	#ifdef VERBOSE
	net.print();
	#endif

	net.randomize();

	#ifdef VERBOSE
	net.print();
	#endif

	net.set(neurons, sinapses);

	#ifdef VERBOSE
	net.print();
	#endif


	Array<double> inputs;
	Array<double> outputs;

	inputs.allocate(layerSizes[0]);
	inputs[0] = 0;
	inputs[1] = 1;
	net.forwardPropagate(inputs);

	#ifdef VERBOSE
	net.print();
	#endif

	outputs.allocate(layerSizes[layerSizes.size() - 1]);
	net.getOutputs(outputs);

	std::cout << "\nTesting forwardPropagate() : " << outputs[0] << " , " << (double)outputs[1];

	if (DOUBLE_EQUALS(outputs[0], 0.916687598) && DOUBLE_EQUALS(outputs[1], 0.9288596354))
	{
		std::cout << " | PASSED" << std::endl;
		return 0;
	}
	else
	{
		std::cout << " | *** FAILED *** (expected 0.916687598 , 0.9288596354)" << std::endl;
		return 1;
	}
}

int RunBackPropagateTests()
{
	Array<size_t> layerSizes(4);
	layerSizes[0] = 2;
	layerSizes[1] = 3;
	layerSizes[2] = 2;
	layerSizes[3] = 2;

	NeuralNet net(layerSizes);

	Array<NeuralNet::Sinapse> sinapses;
	sinapses.allocate(16);
	sinapses[0].weight = 0.1;
	sinapses[1].weight = 0.2;
	sinapses[2].weight = 0.3;
	sinapses[3].weight = 0.4;
	sinapses[4].weight = 0.5;
	sinapses[5].weight = 0.6;
	sinapses[6].weight = 0.7;
	sinapses[7].weight = 0.8;
	sinapses[8].weight = 0.9;
	sinapses[9].weight = 1.0;
	sinapses[10].weight = 1.1;
	sinapses[11].weight = 1.2;
	sinapses[12].weight = 1.3;
	sinapses[13].weight = 1.4;
	sinapses[14].weight = 1.5;
	sinapses[15].weight = 1.6;

	Array<NeuralNet::Neuron> neurons;
	neurons.allocate(9);
	neurons[0].bias = 0;
	neurons[1].bias = 0;
	neurons[2].bias = 0;
	neurons[3].bias = 0;
	neurons[4].bias = 0;
	neurons[5].bias = 0;
	neurons[6].bias = 0;
	neurons[7].bias = 0;
	neurons[8].bias = 0;

	net.set(neurons, sinapses);

	Array<double> inputs;
	Array<double> outputs;

	inputs.allocate(layerSizes[0]);
	inputs[0] = 0;
	inputs[1] = 1;
	
	outputs.allocate(layerSizes[layerSizes.size() - 1]);
	outputs[0] = 1;
	outputs[1] = 0;

	double errorBefore = 0;
	double errorAfter = 0;
	for (int i = 0; i < 1000; ++i)
	{
		errorBefore = net.error(inputs, outputs);
		net.backPropagate(inputs, outputs, 0.5, 0.5);
		errorAfter = net.error(inputs, outputs);
		
		#ifdef VERBOSE
		std::cout << " ERROR_BEFORE: " << errorBefore << " : "
		          << " ERROR_AFTER: " << errorAfter << " : "
		          << (errorAfter - errorBefore) <<  std::endl;
		#endif

		if (errorAfter > errorBefore)
		{
			break;
		}
	}	  

	std::cout.precision(17);
	std::cout << "\nTesting backPropagate() : " 
			  << " ERROR_BEFORE: " << errorBefore << " : "
			  << " ERROR_AFTER: " << errorAfter;

	if (errorBefore > errorAfter)
	{
		std::cout << " | PASSED" << std::endl;
		return 0;
	}
	else
	{
		std::cout << " | *** FAILED *** (error increased)" << std::endl;
		return 1;
	}
}

int RunAllTests()
{
	int failed = 0;

	failed += RunForwardPropagateTests();
	failed += RunBackPropagateTests();

	if (failed > 0)
	{
		std::cout << std::endl << " *** " << failed << " FAILED TESTS *** " << std::endl << std::endl;
	}
	else
	{
		std::cout << std::endl << "ALL TESTS PASSED..." << std::endl << std::endl;
	}

	return failed;
}
