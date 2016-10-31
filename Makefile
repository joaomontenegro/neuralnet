all:
	mkdir -p out
	g++ src/Main.cpp src/NeuralNet.cpp src/Tests.cpp src/TrainIncrement.cpp src/ImageNeuralNet.cpp -g -o out/neural -I./include -lOpenImageIO
