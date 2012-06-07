CC=gcc
CFLAGS= -march=core2 -O2 -pipe -D_XOPEN_SOURCE -Wall -std=c99 -I. `pkg-config --cflags glib-2.0` 
LIBS=`pkg-config --libs glib-2.0`  -lm  -lpthread 
LOW=0
UP=8
UPTAG=-u ${UP}
LOWTAG=-l ${LOW}
CPU=8
DIST=1
KNN=100
#FILE=data/wsj.test1M.lc.500.n0p09.subf.gz.bin
DATAPATH=/scratch/myatbaz/lm/data/
#DATA=wsj.test1M.lc.500.n0p09.subf.gz.bin
#DATA=wsj.test1M.lc.500.n100p0.subf.gz.bin
#DATA=wsj.test1M.lc.500.n100p0.subf.gz.bin.col
#DATA=wsj.test1M.20.n1000p09.subf.gz.bin
DATA=wsj.test1M.lc.20.n100p1.subf.gz.bin
DATAGZ=wsj.test1M.lc.20.n100p1.subf.gz
#DATA=wsj.test1M.lc.20.n100p1.subf.gz.svd100.gz.bin
#TAG=
HOST=$(shell hostname)
dists: dists.c foreach.h heap.o  array.h dists.h
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

run:
#	nohup	./dists ${UPTAG} ${LOWTAG} -p ${CPU} ${SPARSE} -d ${DIST} -f ${DATAPATH}${DATA} > ${DATAPATH}${DATA}.u${UP}d${LOW}.eq${DIST}_${HOST}${TAG} 2>${DATAPATH}${DATA}.u${UP}d${LOW}.eq${DIST}_${HOST}${TAG}.err &
#	./dists ${UPTAG} ${LOWTAG} -p ${CPU} -d$ {DIST} -f ${DATAPATH}${DATA} > ${DATAPATH}foo.out
	zcat  ${DATAPATH}${DATAGZ} | ./preinput.py | ./dists -d ${DIST} -k ${KNN} -p ${CPU} -u ${UP} -l ${LOW} > /scratch/foo.out

heap.o:	heap.c heap.h foreach.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm *.o
