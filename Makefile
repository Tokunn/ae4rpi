all:
	python3 autoencoder.py

clean:
	rm -rf logs/debug

dclean:
	rm -rf __pycache__ logs/debug logs/2019* jobscript/jobscript.sh.o* 

deepclean:
	rm -rf __pycache__ logs/debug logs/2019* jobscript/jobscript.sh.o* npy/*.npy
