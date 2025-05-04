NAME = Main
INCLUDEPATH = ../include
LIBPATH = ../lib

all:
	g++ -Ofast -o ../bin/$(NAME).exe *.cpp -I $(INCLUDEPATH) -L $(LIBPATH)

debug:
	g++ -o ../bin/$(NAME)DEBUG.exe *.cpp -D NDEBUG -I $(INCLUDEPATH) -L $(LIBPATH)
