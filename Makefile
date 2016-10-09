all:
	mkdir -p out
	g++ src/Main.cpp src/NeuralNet.cpp src/Tests.cpp -g -o out/neural -I./include
