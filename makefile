matrixChecker:
	gcc -Wall -Werror -o matrixChecker matrixChecker.c matrixIO.c

matrixAdd:
	mpicc -Wall -Werror -o matrixAdd matrixAdd.c matrixIO.c

matrixGenerator:
	gcc -Wall -Werror -o matrixGenerator matrixGenerator.c matrixIO.c

clean:
	rm matrixGenerator matrixAdd matrixChecker