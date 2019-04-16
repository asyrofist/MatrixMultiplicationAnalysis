import numpy as np
import matplotlib as plt
import seaborn as sns
import random
import pandas as pd
 
class Matrix():
  #Metodo construtor da class Matrix
  def __init__(self, n_rows=16, n_cols=16, min_range = 1, max_range = 100, matrix = None):
    #Python nao permite override de metodos, entao criei o metodo que pode receber uma numpy matrix para entrar na classe ou que seja randomizada e criada uma
    #matriz a partir do numero de linhas e colunas
    if matrix is None:
      #criando uma lista vazia
      matrix = []
      for i in range(0,n_cols):
        tuple = [] #tuplas da matrix
        for j in range(0,n_rows):
          tuple.append(random.randint(min_range, max_range)) #aleatorizando numeros para a matrix e adicionando ao vetor tuple
        matrix.append(tuple) #adicionando vetor recentemente criado a lista matrix
 
      self.matrix = np.matrix(matrix) #criando a matrix como um tipo numpy como atributo da class
      self.n_rows = n_rows #transformando como atributo da class
      self.n_cols = n_cols #transformando como atributo da class
    else:
      self.matrix = np.matrix(matrix)
      self.n_rows = n_rows
      self.n_cols = n_cols
  
  def __repr__(self):
    return str(self.matrix)
  
  @staticmethod
  def multiply(A, B):
    #metodo de multiplicacao ijk ou forca bruta
    if A.n_cols != B.n_rows:
      raise ValueError('Matrices cannot be multiplied n_cols != n_rows') #Nao se pode multiplicar matrizes em que o numero de linhas da primeira matriz seja diferente do numero de colunas da segunda matriz
    else:
      result_matrix = []
      for i in range(0,A.n_rows): #row iterator of Matrix A
        result_row = []
        for k in range(0,B.n_cols): #column iterator of Matrix B
          value = 0
          for j in range(0,A.n_cols): #column iterator of Matrix A and row iterator of Matrix B
            value += int(A.matrix[[i],[j]]) * int(B.matrix[[j],[k]])
          result_row.append(value)
        result_matrix.append(result_row)
      return Matrix(matrix=result_matrix,n_rows=A.n_rows,n_cols=B.n_cols)
    
  @staticmethod
  def arithmetic_operation(A,sign,B):
    if A.n_cols != B.n_cols and A.n_rows != B.n_rows:
      raise ValueError('Matrices cannot be added n_cols != n_rows') 
    
    result_matrix = []
    if sign == '+':
      result_matrix = [[int(A.matrix[[i],[j]]) + int(B.matrix[[i],[j]]) for j in range(0,A.n_cols)] for i in range(0,A.n_rows)] # isso equivale a um loop dentro do outro que gera uma lista
    elif sign == '-':
      result_matrix = [[int(A.matrix[[i],[j]]) - int(B.matrix[[i],[j]]) for j in range(0,A.n_cols)] for i in range(0,A.n_rows)] # isso equivale a um loop dentro do outro que gera uma lista
    
    return Matrix(matrix=result_matrix,n_rows=A.n_rows,n_cols=A.n_cols)
      
  @staticmethod
  def multiply_divide_and_conquer(A,B, MIN_DIMENSION = 4):
    n_r = A.n_rows
    n_c = A.n_cols
 
    #criação da Matriz resultante C
    C = [[0 for i in range(0,n_r)] for j in range(0,n_r)]
 
    a = A.create_sub_matrix(1)
    b = A.create_sub_matrix(2)
    c = A.create_sub_matrix(3)
    d = A.create_sub_matrix(4)
 
    e = B.create_sub_matrix(1)
    f = B.create_sub_matrix(2)
    g = B.create_sub_matrix(3)
    h = B.create_sub_matrix(4)
    
#     if n_r == 1:
#       return Matrix(matrix=[[int(A.matrix[[0],[0]]) * int(B.matrix[[0],[0]])]], n_cols=1,n_rows=1)
      
    if n_r <= MIN_DIMENSION:
        return Matrix.multiply(A,B)
 
    else:
      c11 = Matrix.arithmetic_operation(Matrix.multiply_divide_and_conquer(a,e),'+',Matrix.multiply_divide_and_conquer(b,g))
      c12 = Matrix.arithmetic_operation(Matrix.multiply_divide_and_conquer(a,f),'+',Matrix.multiply_divide_and_conquer(b,h))
      c21 = Matrix.arithmetic_operation(Matrix.multiply_divide_and_conquer(c,e),'+',Matrix.multiply_divide_and_conquer(d,g))
      c22 = Matrix.arithmetic_operation(Matrix.multiply_divide_and_conquer(c,f),'+',Matrix.multiply_divide_and_conquer(d,h))
 
    for i in range(0, n_r//2):
      for j in range(0, n_r//2):
        C[i][j] = int(c11.matrix[[i],[j]])
        C[i][j + n_c//2] = int(c12.matrix[[i],[j]])
        C[i + n_r//2][j] = int(c21.matrix[[i],[j]])
        C[i + n_r//2][j + n_c//2] = int(c22.matrix[[i],[j]])
 
    return Matrix(matrix=C,n_rows=n_r,n_cols=n_c)
  
  @staticmethod
  def multiply_strassen(A, B, MIN_DIMENSION = 4):
    #Algoritmo:
    #|a b|*|e f| = |p5+p4-p2+p6 p1+p2|
    #|c d| |g h|   |p3+p4 p1+p5-p3-p7|
    #  A     B              C
    #
    #p1 = a*(f-h)
    #p2 = (a+b)*h
    #p3 = (c+d)*e
    #p4 = d*(g-e)
    #p5 = (a+d)*(e+h)
    #p6 = (b-d)*(g+h)
    #p7 = (a-c)*(e+f)
    
    #criterio de parada, matriz eh menor que a menor dimensao das matrizes, ou seja o calculo eh muito rapido
    if A.matrix.shape[0] <= MIN_DIMENSION:
      return Matrix.multiply(A,B)
       
    a = A.create_sub_matrix(1)
    b = A.create_sub_matrix(2)
    c = A.create_sub_matrix(3)
    d = A.create_sub_matrix(4)
    
    e = B.create_sub_matrix(1)
    f = B.create_sub_matrix(2)
    g = B.create_sub_matrix(3)
    h = B.create_sub_matrix(4)
    
    p1 = Matrix.multiply_strassen(a,Matrix.arithmetic_operation(f,'-',h)) #p1 = a*(f-h)
    p2 = Matrix.multiply_strassen(Matrix.arithmetic_operation(a,'+',b),h) #p2 = (a+b)*h
    p3 = Matrix.multiply_strassen(Matrix.arithmetic_operation(c,'+',d),e) #p3 = (c+d)*e
    p4 = Matrix.multiply_strassen(d,Matrix.arithmetic_operation(g,'-',e)) #p4 = d*(g-e)
    p5 = Matrix.multiply_strassen(Matrix.arithmetic_operation(a,'+',d),Matrix.arithmetic_operation(e,'+',h)) #p5 = (a+d)*(e+h)
    p6 = Matrix.multiply_strassen(Matrix.arithmetic_operation(b,'-',d),Matrix.arithmetic_operation(g,'+',h)) #p6 = (b-d)*(g+h)
    p7 = Matrix.multiply_strassen(Matrix.arithmetic_operation(a,'-',c),Matrix.arithmetic_operation(e,'+',f)) #p7 = (a-c)*(e+f)
  
    c11 = Matrix.arithmetic_operation(Matrix.arithmetic_operation(Matrix.arithmetic_operation(p5,'+',p4),'-',p2),'+',p6) #p5+p4-p2+p6
    c12 = Matrix.arithmetic_operation(p1,'+',p2) #p1+p2
    c21 = Matrix.arithmetic_operation(p3,'+',p4) #p3+p4
    c22 = Matrix.arithmetic_operation(Matrix.arithmetic_operation(Matrix.arithmetic_operation(p1,'+',p5),'-',p3),'-',p7) #p1+p5-p3-p7
    
    #criação da Matriz resultante C
    C = [[0 for i in range(0,A.n_rows)] for j in range(0,A.n_cols)]
    
    for i in range(0, A.n_rows//2):
        for j in range(0, A.n_cols//2):
          C[i][j] = int(c11.matrix[[i],[j]])
          C[i][j + A.n_cols//2] = int(c12.matrix[[i],[j]])
          C[i + A.n_rows//2][j] = int(c21.matrix[[i],[j]])
          C[i + A.n_rows//2][j + A.n_cols//2] = int(c22.matrix[[i],[j]])
    
    return Matrix(matrix=C,n_rows=A.n_rows,n_cols=A.n_cols)
    
  def create_sub_matrix(self, quadrant):
    #quadrantes:
    #|1 2|
    #|3 4|
    if quadrant == 1:
      sub_matrix = [[int(self.matrix[[i],[j]]) for j in range(0, self.n_cols//2)] for i in range(0, self.n_rows//2)]
    if quadrant == 2:
      sub_matrix = [[int(self.matrix[[i],[j]]) for j in range(self.n_cols//2, self.n_cols)] for i in range(0, self.n_rows//2)]  
    if quadrant == 3:
      sub_matrix = [[int(self.matrix[[i],[j]]) for j in range(0, self.n_cols//2)] for i in range(self.n_rows//2, self.n_rows)]
    if quadrant == 4:
      sub_matrix = [[int(self.matrix[[i],[j]]) for j in range(self.n_cols//2, self.n_cols)] for i in range(self.n_rows//2, self.n_rows)]
    
    return Matrix(matrix=sub_matrix,n_rows=self.n_rows//2,n_cols=self.n_cols//2)
      
  def print_matrix(self):
    #metodo para imprimir a matriz de forma que fique mais simples de entender
    print("MATRIX WITH SHAPE: {}".format(self.matrix.shape))
    print(pd.DataFrame(self.matrix))
    
    
def main(n=8):
    import time
    
    #parametros
    #x_dim = [8,16,32]#,64,128,256,512,1024,2048]#,4096,8192,16384,32768]
    #y_dim = [8,16,32]#,64,128,256,512,1024,2048]#,4096,8192,16384,32768]
    x = n
    experiment_times = []
    
    
    A = Matrix(x, x)
    B = Matrix(x, x)
    
    #start = time.time()  
    #Matrix.multiply_strassen(A,B)
    #end = time.time()
    
    #obj_dict = {'Algorithm':'Strassen','dimension':x,'execution_time':(end-start)}
    #experiment_times.append(obj_dict)
    
    start = time.time()
    Matrix.multiply(A,B)
    end = time.time()
    
    obj_dict = {'Algorithm':'ijk','dimension':x,'execution_time':(end-start)}
    experiment_times.append(obj_dict)
    
    #start = time.time()
    #Matrix.multiply_divide_and_conquer(A,B)
    #end = time.time()
    
    #obj_dict = {'Algorithm':'divide-and-conquer','dimension':x,'execution_time':(end-start)}
    #experiment_times.append(obj_dict)
      
  
    results = pd.DataFrame.from_dict(experiment_times)
    
    results.to_csv('/mnt/results/run_strassen{}.csv'.format(time.time()),index=False)
 
 
if __name__ == '__main__':
    main()