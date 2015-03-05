CC = g++
LINK = g++
INSTALL = install
CFLAGS = -Wall `pkg-config opencv --cflags` -I /usr/include/boost-1_46 -I.
LFLAGS = -Wall /usr/lib/libgtest.a /usr/lib/libgtest_main.a `pkg-config opencv --libs` -L /usr/lib  -lboost_system -lboost_filesystem -lopencv_features2d -lopencv_nonfree -lopencv_ocl -lopencv_highgui -lopencv_core -lpthread
#OBJS = car.o cartracker.o

all: cartracker test



cartracker.o: cartracker.cpp car.h
	$(CC) $(CFLAGS) -c $^ 

car.o: car.h


cartracker: cartracker.o car.o
	$(LINK) $^ -o $@  $(LFLAGS)

test.o: test.cpp car.h
	$(CC) $(CFLAGS) -c $^ 

test: test.o car.o
	$(LINK) $^ -o $@  $(LFLAGS)


clean:
	rm -f  *.o

install:
	cp cartracker /usr/bin/

