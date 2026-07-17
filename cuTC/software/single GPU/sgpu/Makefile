CC = gcc

NVCC = nvcc

CFLAGS = -O3   -std=c99 -march=native -D_GNU_SOURCE -DCISS_DEBUG
#NCFLAGS = -O2  -G -arch=sm_86 -D ALSAS_DEBUG
NCFLAGS = -O2  -arch=sm_86
#-D ALS_DEBUG


TARGET = tc

$(TARGET):tc.o base.o io.o util.o sptensor.o matrixprocess.o ciss.o loss.o ccd.o als.o sgd.o completion.o sampling_tns.o
	$(CC) $(CFLAGS) $^ -o $@ -L/usr/local/cuda/lib64 -lcudart -lcuda -lcusolver -lm

tc.o: tc.c
	$(CC) -o $@ -c $(CFLAGS) $<
base.o: base.c
	$(CC) -o $@ -c $(CFLAGS) $<
#csf.o: csf.c
#	$(CC) -o $@ -c $(CFLAGS) $<
util.o:util.c
	$(CC) -o $@ -c $(CFLAGS) $<
matrixprocess.o:matrixprocess.c 
	$(CC) -o $@ -c $(CFLAGS) $<  
io.o: io.c 
	$(CC) -o $@ -c $(CFLAGS) $<
sptensor.o: sptensor.c
	$(CC) -o $@ -c $(CFLAGS) $<
ciss.o: ciss.c
	$(CC) -o $@ -c $(CFLAGS) $<
completion.o: completion.c als.cu ccd.cu sgd.cu
	$(CC) -o $@ -c $(CFLAGS) $<
loss.o: loss.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda/lib64 -lcudart -lcuda
als.o: als.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda/lib64 -lcudart -lcuda -lcusolver -lm
sgd.o: sgd.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda/lib64 -lcudart -lcuda
ccd.o: ccd.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda/lib64 -lcudart -lcuda
sampling_tns.o: sampling_tns.c sampling_tns.h
	$(CC) -o sampling_tns.o -c $(CFLAGS) sampling_tns.c

#cpd.o: cpd.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#matrixnormal_host.o: matrixnormal_host.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#matrixnormal_slave.o: matrixnormal_slave.c
#	$(CC) -o $@ -c $(CFLAG_SLAVE) $<
#mttkrp.o: mttkrp.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#mttkrp_slave.o:mttkrp_slave.c
#	$(CC) -o $@ -c $(CFLAG_SLAVE) $<
clean:
	rm *.o $(TARGET)
LDLIBS +=
LDLIBS += -lm
