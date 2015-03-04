CC = g++
LINK = g++
INSTALL = install
CFLAGS = -Wall `pkg-config opencv --cflags` -I /usr/include/boost-1_46 -I.
LFLAGS = -Wall `pkg-config opencv --libs` -L /usr/lib  -lboost_system -lboost_filesystem -lopencv_features2d -lopencv_nonfree -lopencv_ocl -lopencv_highgui -lopencv_core
#OBJS = car.o cartracker.o

all: cartracker



cartracker.o: cartracker.cpp car.h
	$(CC) $(CFLAGS) -c $^ 

car.o: car.h
#	$(CC) $(CFLAGS) -o $@ -c $^

cartracker: cartracker.o car.o
	$(LINK) $^ -o $@  $(LFLAGS)


clean:
	rm -f  *.o

install:
	cp cartracker /usr/bin/

