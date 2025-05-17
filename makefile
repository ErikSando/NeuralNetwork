NAME = Main

all:
	g++ -Ofast -o bin/$(NAME).exe src/*.cpp -D NDEBUG -I include -L lib

debug:
	g++ -o bin/$(NAME)DEBUG.exe src/*.cpp -I include -L lib
