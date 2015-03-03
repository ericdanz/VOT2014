CC = g++
LINK = g++
INSTALL = install
CFLAGS = `pkg-config opencv --cflags` -I /usr/include/boost-1_46 -I.
LFLAGS = `pkg-config opencv --libs` -L /usr/lib  -lboost_system -lboost_filesystem -lopencv_features2d -lopencv_nonfree -lopencv_ocl -lopencv_highgui -lopencv_core
all: surftest

surftest.o: SURFTutorial.cpp
	$(CC) $(CFLAGS) -o $@ -c $^

surftest: surftest.o
	$(LINK) -o $@ $^ $(LFLAGS)

clean:
	rm -f  *.o

install:
	cp surftest /usr/bin/
