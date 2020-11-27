import pandas as pd
import numpy as np
import math
import random
import copy

# Esta función recive el nombre del archivo CSV y devuelve un dataframe de Pandas
def leer_dataset(nom_archivo):
    return pd.read_csv(nom_archivo)
# Retorna la lista de valores únicos de una columna de un dataframe
def cardinalidad(df, X):
    return df[X].unique()
#retorna el número de instancias de un dataframe que satisface una determinada condición.
def count(dataframe, index_list, values_list):
    resultado = dataframe
    for i in range(len(index_list)):
        #Se obtiene el nombre de la columna y el valor
        index = index_list[i]
        value = values_list[i]

        #Se filtra las filas que cumplen con la condicion de esa columna
        resultado = resultado[resultado[index] == value]
    #Devuelve el numero de instancias
    c =resultado.shape[0]
    return c

# Esta funcióm devuelve un reporte del factor que contiene a la variable X
def estimar_marginal(df,X, alpha):
    card = cardinalidad(df,X) # lista de valores únicos

    nfilas = df.shape[0]

    probList = []
    for valor in card:
        m = count(df,[X],[valor])
        p = (m + alpha)/(nfilas + len(card)*alpha)
        probList.append(p)
    probColName = "P"
    columns = {X:card, probColName: probList}

    return pd.DataFrame(columns)

def estimar_conjunta(df,variables,alpha):
    card = {}
    nfilas = 1

    ## filas en el df
    n = df.shape[0]

    for var in variables:
        card[var] = cardinalidad(df,var)
        nfilas = nfilas*len(card[var])

    columns = {}

    n_rep = 1
    for var, cardinal in card.items():
        columns[var] = [ cardinal[int(i/n_rep)%len(cardinal)] for i in range(nfilas) ]
        n_rep = n_rep * len(cardinal)

    result = pd.DataFrame(columns)

    probList = []

    #Iteramos por cada fila del dataframe y hallamos la probabilidad condicional
    for i in result.index:
        valores = []
        for var in variables:
            valores.append(result[var][i])

        n_intersection = count(df,variables,valores)

        p = (n_intersection + alpha)/ ( n + n_rep*alpha)

        probList.append(p)

    result['P'] = probList
    return result

def estimar_condicional(df, X, Y, alpha):
    # hallar el número total de filas 
    nFilasResult = 1
    nfilas = df.shape[0]
    cardTarget = cardinalidad(df,X)
    nFilasResult = nFilasResult*len(cardTarget)
    
    cardY = {}
    for var in Y:
        cardY[var] = cardinalidad(df,var)
        nFilasResult = nFilasResult*len(cardY[var])
    
    columns = {}
    
    n_rep = 1
    
    colTarget = [cardTarget[i%len(cardTarget)] for i in range(nFilasResult)]
    
    columns[X] = colTarget
    n_rep = n_rep * len(cardTarget)
    
    for var, card in cardY.items():
        columns[var] = [ card[int(i/n_rep)%len(card)] for i in range(nFilasResult) ]
        
        n_rep = n_rep * len(card)

    
    result = pd.DataFrame(columns)
    probList = []
    
    prodCardY = 1
    for var in Y:
        prodCardY = prodCardY* len(cardY[var])
        
    #Iteramos por cada fila del dataframe y hallamos la probabilidad condicional
    for i in result.index:
        valTarget = result[X][i]
    
        valPadres = []
        for var in Y:
            valPadres.append(result[var][i])
      
        totalVar = [i for i in Y]
        totalVar.append(X)
        totalVal = valPadres
        totalVal.append(valTarget)
        
        
        prob_X_Y = (count(df,totalVar,totalVal) + alpha) / (nfilas + prodCardY*len(cardTarget)*alpha) 
        
        prob_Y = (count(df,Y,valPadres) + alpha) / (nfilas + prodCardY*alpha)
        p = prob_X_Y/prob_Y
        probList.append(p)
    
    # normalizamos las probabilidades
    for i in range(int (nFilasResult / len(cardTarget))):
        suma = 0
        for j in range(len(cardTarget)):
            suma = suma + probList[i*len(cardTarget) + j]
        
        for j in range(len(cardTarget)):
            probList[i*len(cardTarget) + j] = probList[i*len(cardTarget) + j] / suma 
    
    result['P'] = probList
    return result

class Preprocessor:
    def __init__(self, df ,alpha):
        self.variables = self.get_vars(df)
        self.alpha = alpha
        self.df = df
        self.card = self.get_cards(df)
        self.marg_cond = self.preprocess(df)
        self.conj = self.calcular_conjunta(df)
        self.instancias = len(df.index)

    def get_cards(self, df):
        nodes = self.variables
        card = {}
        for value in nodes:
            card[value] = df[value].unique()
        return card

    def get_vars(self,df):
        variables = []
        for i in df:
            variables.append(i)

        return variables

    def calcular_conjunta(self,df):
        prob = {}

        #calcular las conjunta de cada par de variables diferentes
        for variable_1 in self.variables:
            for variable_2 in self.variables:
                if (variable_1 == variable_2):
                    continue
                prob[tuple([variable_1, variable_2])] = estimar_conjunta(df,[variable_1,variable_2], self.alpha)
        return prob

    #preprocesa las marginales y las condicionales para cada para de variables
    def preprocess(self,df):
        prob = {}

        #calcular las marginales de todas la variables
        for variable in self.variables:
            prob[tuple(variable)] = estimar_marginal(df,variable,self.alpha)

        #calcular las condicionales de cada par de variables diferentes
        for variable_1 in self.variables:
            for variable_2 in self.variables:
                if (variable_1 == variable_2):
                    continue
                prob[tuple([variable_1, variable_2])] = estimar_condicional(df,variable_1, [variable_2], self.alpha)
        return prob


    def get_preprocess_marginal(self, variable, valor):
        resultado = self.marg_cond[(variable,)]
        resultado = resultado[resultado[variable]==valor]
        return list(resultado['P'])[0]

    def get_preprocess_conjunta(self, variables, valores):
        resultado = self.conj[variables]
        for i in range(2):
            index = variables[i]
            value = valores[i]
            #filtrar la columna
            resultado = resultado[resultado[index] == value]

        return list(resultado['P'])[0]

    def get_preprocess_condicional(self, factor, valores):
        resultado = self.marg_cond[factor]
        for i in range(2):
            index = factor[i]
            value = valores[i]
            #filtrar la columna
            resultado = resultado[resultado[index] == value]
        return list(resultado['P'])[0]

# Retorna los padres de los nodos
def get_parents(grafo):
    result = {}
    
    for node, list_adyacentes in grafo.items():
        if node not in result:
            result[node] = set()
        
        if (len(list_adyacentes) > 0):  
            for child in list_adyacentes:
                if child not in result:
                    result[child] = set()
                    result[child].add(node)
                else:
                    result[child].add(node)
    
    return result

class Red_Bayesiana:
    def __init__(self, df, grafo, alpha, preprocessor = None):
        self.alpha = alpha
        self.estructura = grafo
        self.parents = get_parents(self.estructura)
        self.factores = self.generate(df)
        print("hola")
        self.card = self.get_cards(df)
        self.preprocessor = preprocessor
        
    def get_cards(self, df):
        nodes = self.estructura.keys()
        card = {}
        for value in nodes:
            card[value] = df[value].unique()
        return card
    #Genera todos los factores de acuerdo a la estructura
    def generate(self, df):
        factores = {}
        parents = self.parents
        for child, parent_list in parents.items():
            if (len(parent_list) == 0):
                print(child)
                factores[tuple([child])] = estimar_marginal(df,child, self.alpha)
            else:
                key = list(parent_list)
                key.insert(0, child)
                key = tuple(key)
                factores[key] = estimar_condicional(df,child, list(parent_list), self.alpha)
        return factores
    
    def printFactores(self):
        print("Factores del grafo")
        for key, df in self.factores.items():
            
            if (len(key)==1):
                print ("P(" + key[0] + ")", end=" ")
            else:
                s = None
                for i in range(len(key)):
                    if (i==0):
                        s = "P(" + key[0] + "|"
                    elif(i == len(key) - 1):
                        s = s + key[i] + ")"
                    else:
                        s = s + key[i] + ","
                print (s, end=" ")
                
    #Función para calcular todos los ancestros de un nodo
    def get_all_ancestor(self, current, current_parents, amount_of_ancestors, visited):
        visited[current] = True
        for parent in current_parents:
            if(not visited[parent]):
                self.get_all_ancestor(parent, self.parents[parent], amount_of_ancestors, visited)
            amount_of_ancestors[current] = amount_of_ancestors[current] + amount_of_ancestors[parent] + 1
        
    #Devuelve las variables ocultas ordenados en orden topológico
    def get_hidden_variables(self, consulta, evidencia):
        nodes = self.estructura.keys()
        amount_of_ancestors = {}
        visited = {}
        
        for node in nodes:
            visited[node] = False
            amount_of_ancestors[node] = 0
        
        for node in nodes:
            if(not visited[node]):
                self.get_all_ancestor(node, self.parents[node], amount_of_ancestors, visited)
        
        hidden_variables = []
        for variable in nodes:
            if ((variable not in evidencia) and variable != consulta):
                #Cada variable esta asociada a la cantidad de ancestros
                hidden_variables.append([variable, amount_of_ancestors[variable]])
        #Se ordena de menor a mayor de acuerdo a la cantidad de ancestros
        #Esto nos garantiza que el nodo que tiene padre va estar después de su padre
        hidden_variables = sorted(hidden_variables, key=lambda item: item[1])
        
        Z = []
        for i in hidden_variables:
            Z.append(i[0])
        return Z
    #El valor de la distribuicion marginal
    def estimar_marginal(self, factor, valores): 
        assert len(valores) == 1, "Marginal solo acepta un valor"
        #Se filtra el factor
        resultado = self.factores[factor]
        resultado = resultado[resultado[factor[0]]==valores[0]]
        return list(resultado['P'])[0]
    #El valor de la distribuicion condicional
    def estimar_condicional(self, factor, valores):
        #Se filtra el factor
        resultado = self.factores[factor]
        for i in range(len(factor)):
            index = factor[i]
            value = valores[i]
            #filtrar la columna 
            resultado = resultado[resultado[index] == value]
        return list(resultado['P'])[0]

def inferencia(clase, e_variables , e_values, red_bayesiana):
    
    var_val = {}
    
    for i in range(len(e_variables)):
        var_val[e_variables[i]] = e_values[i]
    
    # retorna un diccionario con llaves los nodos y valores 
    parentsList = red_bayesiana.parents
    
    #Cardinalidad de la variable clase
    card = red_bayesiana.card[clase]
    list_prob = []
    
    for valor in card:
        p = 1
        var_val[clase] = valor
        for var, parents in parentsList.items():
            if (len(parents) == 0):
                p = p * red_bayesiana.estimar_marginal((var,),[var_val[var]] )
             
            else:
                
                values = [var_val[var]]
                
                for parent in parentsList[var]:
                    values.append(var_val[parent])

                key = list(parentsList[var])
                key.insert(0,var)
                key = tuple(key)
                p = p * red_bayesiana.estimar_condicional(key,values)
      
        list_prob.append(p)
    
    columns = {}
    
    columns[clase] = card
    columns['P'] = list_prob
    
    return pd.DataFrame(columns)

def get_scope(factores, variable):
    factores_with_variable = []
    factores_without_variable = []
    for factor in factores:
        if(variable in factor):
            factores_with_variable.append(factor)
        else:
            factores_without_variable.append(factor)
    return (factores_with_variable, factores_without_variable)

def get_vars(phi_prima):
    involved = []
  
    for element in phi_prima:
        for value in element:
            if(value not in involved):
                involved.append(value)
  
    return involved

def sum_product_eliminate_var(factores, variable_to_eliminate, consulta, valor_consulta, evidencia, valor_evidencia , red_bayesiana ):
    # factores que se van a factorizar
    
    phi_prima, phi_prima_prima = get_scope(factores,variable_to_eliminate)
      
    # lista de variables involucradas en phi_prima
    variables_in_scope = get_vars(phi_prima)

    #diccionario: llave -> variable , valor -> lista de cardinalidad
    var_to_values = {}
    
    for var in variables_in_scope:
        if (var != consulta and (var not in evidencia)):
            var_to_values[var] = red_bayesiana.card[var]
    var_to_values[consulta] = [valor_consulta]
    
    for i in range(len(evidencia)):
        var_to_values[evidencia[i]] = [valor_evidencia[i]]
    
    # llave: variable , valor --> indice en una n_upla
    var_to_index = {}

    #total number of posibilites
    n =  1
    for var, val in var_to_values.items():
        n = n*len(val)
  
    posibilidades =  []
    for i in range(n):
        posibilidades.append([])
        
    groups = 1
    pos = 0
    for var in variables_in_scope:
        var_to_index[var] = pos
        pos = pos + 1
        values = var_to_values[var]
        m = len(values)
        groups = groups*m
        rep = int(n/groups)
        for i in range(groups):
            for j in range(rep):
                posibilidades[i*rep + j].extend([values[i%m]])
    # factores que quedan
  
    p = 0 
    # iterar entre todas las posibilidades
    for n_upla in posibilidades:
    
        p_local = 1
        for dist in phi_prima:
            var = dist[0] # El nodo
            cond = dist[1:] # Los padres del nodo
            
            # hallar los valores que tomas las variables en la distribuiciones
            if (len(cond) == 0):
                p_marginal = red_bayesiana.estimar_marginal(dist,[n_upla[var_to_index[var]]])
                p_local = p_local*p_marginal
            else:
                valores = []
                for var in dist:
                    index = var_to_index[var]
                    valores.append(n_upla[index])
                
                p_condicional = red_bayesiana.estimar_condicional(dist, valores)
              
                p_local = p_local*p_condicional
        # sumatoria de los valores en los factores dentro del get_scope
        p = p + p_local
        
    return [phi_prima_prima, p]

def sum_product(factores, consulta, valor_consulta, evidencia, valor_evidencia, red_bayesiana):
    
    var_to_val = {}
    for i in range(len(evidencia)):
        var_to_val[evidencia[i]] = valor_evidencia[i]
    
    var_to_val[consulta] = valor_consulta
    
    p = 1
    
    for dist in factores:
        var = dist[0]
        cond = dist[1:]
        if (len(cond) == 0):
            p_marginal = red_bayesiana.estimar_marginal(dist,[var_to_val[var]])
            p = p*p_marginal
        else:
            valores = []
            for var in dist:
                valores.append(var_to_val[var])
         

            p_condicional = red_bayesiana.estimar_condicional(dist, valores)
           
            p = p*p_condicional
    
    return p

def sum_product_ve(consulta, valor_consulta , evidencia, valor_evidencia, red_bayesiana):
    Z = red_bayesiana.get_hidden_variables(consulta, evidencia)
  
    tetha = []
    factores = red_bayesiana.factores.keys() # lista de factores
    for i in range(len(Z)):
  
        
        if(len(factores) == 0):
            break
        [nuevos_factores,sumatoria] =  sum_product_eliminate_var(factores, Z[i], consulta, 
                                                                 valor_consulta, evidencia, 
                                                                 valor_evidencia, red_bayesiana)
    
        factores = nuevos_factores
        tetha.append(sumatoria)
    
    if(len(factores)>0):
        p_last = sum_product(factores,consulta,  valor_consulta, evidencia,valor_evidencia, red_bayesiana)
        tetha.append(p_last)
    p = 1
    for sumatoria in tetha:
        p = p*sumatoria
    
    return p

def inferencia_con_variables_ocultas(consulta, evidencia, valor_evidencia, red_bayesiana):
    
    card = red_bayesiana.card[consulta]
    list_prob = []
    for valor_consulta in card:
        p = sum_product_ve(consulta, valor_consulta , evidencia, valor_evidencia, red_bayesiana)
        list_prob.append(p)
    columns = {}
    columns[consulta] = card
    columns['P'] = list_prob
    
    return pd.DataFrame(columns)

class Structure:
    def __init__(self, list_adj ,preprocessor = None):
        self.estructura = copy.deepcopy(list_adj)
        self.parents = get_parents(self.estructura)
        self.set_edge = self.get_set_edge(list_adj)
        self.n_nodes = len(list_adj.keys())
        self.n_edges = len(self.set_edge)
        self.nodes_indexes = self.get_indexes()
        self.graph_id = self.get_graph_id()
        self.edges_not_graph = self.get_edges_not_graph()
        self.preprocessor = preprocessor

    def get_indexes(self):
        indexes = {}
        i = 0
        for node in self.estructura.keys():
            indexes[node] = i
            i = i + 1
        return indexes

    def get_graph_id(self):
        VxV = self.n_nodes * self.n_nodes
        g_id = [0] * (VxV)
        for node, list_adj in self.estructura.items():
            for neighbor in list_adj:
                index = self.n_nodes * self.nodes_indexes[node] + self.nodes_indexes[neighbor]
                g_id[index] = 1
        return g_id

    def get_set_edge(self, list_adj):
        result = set()
        for v, list_neighbors in list_adj.items():
            for neighbor in list_neighbors:
                result.add((v,neighbor))
        return result

    def get_edges_not_graph(self):
        edges_possibles = set()
        i = 0
        j = self.n_nodes - 1
        k = 0
        nodes = list(self.nodes_indexes.keys())
        for index in range(len(self.graph_id)):
            if(index != i):
                if(self.graph_id[index] == 0):
                    edges_possibles.add((nodes[i%self.n_nodes], nodes[k%self.n_nodes]))
            if(j == index):
                i = i + self.n_nodes + 1
                j = j + self.n_nodes

            k = k + 1

        return edges_possibles

    def add_edge(self,v1,v2):
        edge = (v1, v2)
        self.edges_not_graph.remove(edge)
        self.set_edge.add(edge)

        self.estructura[v1].append(v2)
        self.parents[v2].add(v1)
        self.n_edges = self.n_edges + 1

        index = self.n_nodes*self.nodes_indexes[v1] + self.nodes_indexes[v2]
        self.graph_id[index] = 1 # this means that a edge was added

    def operator_for_eliminate_edge(self):
        if (self.n_edges == 0):
            return
        edge = random.choice(list(self.set_edge))
        self.set_edge.remove(edge)
        self.edges_not_graph.add(edge)

        v1 = edge[0]
        v2 = edge[1]
        self.estructura[v1].remove(v2)
        self.parents[v2].remove(v1)
        self.n_edges = self.n_edges - 1

        #Update graph id
        index = self.n_nodes*self.nodes_indexes[v1] + self.nodes_indexes[v2]
        self.graph_id[index] = 0 # this means that a edge was removed
        return edge

    def operator_for_add_edge(self):
        total_edges = (self.n_nodes) * (self.n_nodes - 1)
        if(total_edges == len(self.set_edge)):
            return

        edge = random.choice(list(self.edges_not_graph))
        self.edges_not_graph.remove(edge)
        self.set_edge.add(edge)
        v1 = edge[0]
        v2 = edge[1]
        self.estructura[v1].append(v2)
        self.parents[v2].add(v1)

        index = self.n_nodes*self.nodes_indexes[v1] + self.nodes_indexes[v2]
        self.graph_id[index] = 1 # this means that a edge was added

        self.n_edges = self.n_edges + 1

    def operator_for_reverse_edge(self):
        total_edges = (self.n_nodes) * (self.n_nodes - 1)
        if (self.n_edges == 0 or self.n_edges == total_edges):
            return

        possible = False
        inverted_edge = None
        edge_to_remove = None
        for graph_edge in self.set_edge:
            inverted = (graph_edge[1], graph_edge[0])
            if (not (inverted in self.set_edge)):
                edge_to_remove = graph_edge
                inverted_edge = inverted
                possible = True
                break

        if(not possible):
            return

        self.set_edge.remove(edge_to_remove)
        self.edges_not_graph.add(edge_to_remove)

        v1 = edge_to_remove[0]
        v2 = edge_to_remove[1]
        self.estructura[v1].remove(v2)
        self.parents[v2].remove(v1)

        v1 = inverted_edge[0]
        v2 = inverted_edge[1]
        self.estructura[v1].append(v2)
        self.parents[v2].add(v1)

        self.set_edge.add(inverted_edge)
        self.edges_not_graph.remove(inverted_edge)

        index = self.n_nodes*self.nodes_indexes[v1] + self.nodes_indexes[v2]
        self.graph_id[index] = 1 # this means that a edge was added
        index = self.n_nodes*self.nodes_indexes[v2] + self.nodes_indexes[v1]
        self.graph_id[index] = 0 # this means that a edge was removed

    def is_cyclic_util(self, v, visited, recStack):
        visited[v] = True
        recStack[v] = True

        for neighbour in self.estructura[v]:
            if(not visited[neighbour]):
                if (self.is_cyclic_util(neighbour, visited, recStack)):
                    return True
            elif (recStack[neighbour]):
                return True

        recStack[v] = False
        return False

    def is_cyclic(self):
        nodes = self.estructura.keys()
        visited = {}
        recStack = {}
        for node in nodes:
            visited[node] = False
            recStack[node] = False

        for node in nodes:
            if (not visited[node]):
                if self.is_cyclic_util(node, visited, recStack):
                    return True

        return False

def get_entropia(estructura_red):
    ## Obtenemos la cardinalidad de cada variable
    variables = estructura_red.estructura.keys()
    n = len(variables)
    # diccionario de variable -> lista de valores

    r = {}
    for var in variables:
        r[var] = estructura_red.preprocessor.card[var]


    # diccionario de variables -> lista de padres
    q = estructura_red.parents
    #TO DO

    # N total de instancias
    N = estructura_red.preprocessor.instancias
    result = 0

    # i = 1 ... n
    # xi
    for variable in variables:
        xi = variable
        pa_xi = list(q[variable])

        # si la variable no tiene padre
        if(len(pa_xi)==0):
            for valor in r[variable]:

                # Ej: P(A = 0).log2(P(A=0))
                prob_xi = estructura_red.preprocessor.get_preprocess_marginal(xi, valor)
                result = result + prob_xi*math.log(prob_xi,2)

            continue

        # j = 1 .... qi
        pa_xi = pa_xi[0]
        for parent_val in r[pa_xi]:
            # k = 1... ri
            for valor in r[variable]:
                # Calculo de P(xi|pa(xi))
                p_xi_paxi = estructura_red.preprocessor.get_preprocess_condicional((xi,pa_xi),[valor,parent_val])
                p_conj_xi_pxi = estructura_red.preprocessor.get_preprocess_conjunta((xi,pa_xi),[valor,parent_val])

                result = result + p_conj_xi_pxi*(math.log(p_xi_paxi,2))

    result = result*(-1)*N

    return result

def get_number_independent_parameters(estructura_red):
    cardinalities = estructura_red.preprocessor.card
    parents = estructura_red.parents

    sum = 0

    for r in cardinalities.keys():
        q = 1
        for parent in parents[r]:
            q = q * len(cardinalities[parent])
        sum = sum + (len(cardinalities[r]) - 1)*q

    return sum

def get_akaike(estructura_red):
    #
    K = get_number_independent_parameters(estructura_red)
    e = get_entropia(estructura_red)

    return e + K

def get_mdl(estructura_red):
    N = estructura_red.preprocessor.instancias
    K = get_number_independent_parameters(estructura_red)
    e = get_entropia(estructura_red)

    return e + K/2*math.log(N,2)

def get_k2(estructura_red):
    # Obtenemos la cardinalidad de cada variable
    variables = estructura_red.preprocessor.variables
    n = len(variables)
    df = estructura_red.preprocessor.df
    q = estructura_red.parents

    r = {}
    for var in variables:
        r[var] = estructura_red.preprocessor.card[var]

    result = 1

    for variable in variables:
        xi = variable
        pa_xi = list(q[variable])
        if(len(pa_xi)==0):
            continue

        ri = len(r[variable])

        # j = 1 .... qi
        pa_xi = pa_xi[0]
        for parent_val in r[pa_xi]:

            # Nij número de instancias donde Pa(xi) toma su j-esimo valor
            Nij = count(df, [pa_xi], [parent_val])


            a = math.factorial(ri -1) / math.factorial(ri-1 + Nij)

            result = result*a
            # k = 1... ri
            for valor in r[variable]:

                # Nijk número de instancias donde Pa(xi) toma su j-esimo valor y xi toma su k-esimo valor
                Nijk = count(df, [pa_xi,xi],[parent_val,valor])
                result =  result* math.factorial(Nijk)

    return result

def greedy_local_search_modified(initial_solution, score_function, set_operators = [0, 1, 2]):
    best_solution = initial_solution
    Progress = True
    iteraciones = 0
    while(Progress):
        solution = best_solution
        Progress = False
        for operator in set_operators:
            candidate_solution = copy.deepcopy(solution)
            if(operator == 0):
                candidate_solution.operator_for_add_edge()
            elif(operator == 1):
                candidate_solution.operator_for_eliminate_edge()
            else:
                candidate_solution.operator_for_reverse_edge()

            iteraciones = iteraciones + 1
            if (not candidate_solution.is_cyclic()):
                if(score_function(candidate_solution) > score_function(best_solution)):
                    best_solution = copy.deepcopy(candidate_solution)
                    Progress = True
    return (best_solution, iteraciones)

def greedy_algorithm_search_space(initial_graph, score_function):
    N = initial_graph.n_nodes
    MAX_ITERACIONES = 2**(N*(N-1)) * 0.25
    print(MAX_ITERACIONES)
    MAX = 5
    graph = copy.deepcopy(initial_graph)
    g_id = initial_graph.graph_id
    same_id = 0
    current_iteration = 0
    while(current_iteration < MAX_ITERACIONES):
        while(current_iteration < MAX_ITERACIONES and same_id < MAX):
            graph, iteraciones = greedy_local_search_modified(graph, score_function, [0, 1, 2])
            current_iteration = current_iteration + iteraciones
            print("*******************")
            print(graph.estructura)
            print(current_iteration)
            print("*******************")
            if(graph.graph_id == g_id):
                same_id = same_id + 1
            else:
                g_id = graph.graph_id
                same_id = 0

        if(current_iteration < MAX_ITERACIONES):
            new_graph = graph
            candidate_graph = copy.deepcopy(new_graph)
            cant = 3
            op = random.randint(0,2)
            if(op == 0):
                for i in range(cant):
                    candidate_graph.operator_for_add_edge()
                    if(not candidate_graph.is_cyclic()):
                        new_graph = candidate_graph
                    candidate_graph = copy.deepcopy(new_graph)

            elif(op == 1):
                for i in range(cant):
                    graph.operator_for_eliminate_edge()
                    if(not candidate_graph.is_cyclic()):
                        new_graph = candidate_graph
                    candidate_graph = copy.deepcopy(new_graph)
            else:
                for i in range(cant):
                    graph.operator_for_reverse_edge()
                    if(not candidate_graph.is_cyclic()):
                        new_graph = candidate_graph
                    candidate_graph = copy.deepcopy(new_graph)
            print("nuevo grafo")
            print(graph.estructura)
            same_id = 0
            current_iteration = current_iteration + cant
            graph = new_graph


    return current_iteration
