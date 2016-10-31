#ifndef _IMAGE_NEURAL_NET_H_
#define _IMAGE_NEURAL_NET_H_

#include "NeuralNet.h"

bool createImageNeuralNet(std::string imagePath, 
						  Array<size_t>& neuronsPerHiddenLayer,
						  size_t numOutputs,
						  NeuralNet* outNet);

bool getPixels(std::string imagePath, Array<double>& pixels);

bool forwardPropImageNet(NeuralNet* net, Array<double>);

#endif
