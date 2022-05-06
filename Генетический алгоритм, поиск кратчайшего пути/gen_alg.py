# -*- coding: utf-8 -*-

#Решение задачи поиска кратчайшего пути применяя генетический алгоритм
#Работу выполнил Михеев Дмитрий Васильевич

import networkx as nx
from pyvis import network as net
import matplotlib.pyplot as plt
import numpy as np
import random

count = 10 # - Количество особей в поколении
value_max = 727 # - Предел популяций
start_node = 0 # - Номер начальной клетки
end_node = 4 # - Номер конечной клетки

#Генерация рандомных чисел в виде таблички
matrix = np.zeros([count, count])
for i in np.arange(0, count, 1):
    for j in np.arange(0, count, 1):
        if i != j:
            matrix[i,j] = int(random.randint(0, value_max))
            matrix[j,i] = matrix[i,j]
        else:
            matrix[i,j] = int(0)

for x in range(len(matrix)):
    for y in range(len(matrix)):
        print('%5d' % matrix[x][y], end=' ')
    print('\n')

#Использование таблички рандомных чисел для построения графа маршрутов
graph_nx = nx.Graph()
for i in np.arange(0, len(matrix), 1):
    for j in np.arange(0, len(matrix), 1):
        graph_nx.add_edge(i, j, weight=float(matrix[i][j]))

layout = nx.spring_layout(graph_nx)
labels = nx.get_edge_attributes(graph_nx, 'weight')
nx.draw_networkx_edge_labels(graph_nx, pos=layout, edge_labels=labels)
nx.draw(graph_nx, layout, with_labels=True, node_size=1000,font_size=20)

#Создание хромосомы
def gen_make(start_node, end_node, gen_len, graph_nx):
    genes = []
    genes.append(start_node)
    i = 1
    while i < gen_len-1:
        new_gen = random.choice([n for n in graph_nx.neighbors(start_node)])
        if (new_gen in genes) or (new_gen == end_node):
            continue
        else:
            genes.append(new_gen)
            i += 1
    genes.append(end_node)
    return genes

#Создание популяции
def popul_make(start_node, end_node, gen_len, graph_nx):
    population = []
    for i in range(n_chrom):
        population.append(gen_make(start_node, end_node, gen_len, graph_nx))
    return population

#Турнирная селекция (Один  из методов эволюции популяции, при котором выбираются две особи 
#и сравниваются по наиболее подходящему значению)
def tournament_selection(matrix, population, n_chrom):
    population_selected = []
    #количество турниров определяется количеством особей в популяции
    for lap in range(n_chrom):
        chrom_1 = random.choice(population)
        chrom_2 = random.choice(population)
        if path_weight(matrix, chrom_1) <= path_weight(matrix, chrom_2):
            population_selected.append(chrom_1)
        else:
            population_selected.append(chrom_2)
    return population_selected

#Расчет длины предложенного особью пути
def path_weight(matrix, chrom):
    weight = 0
    for gen in range(len(chrom) - 1):
        weight += matrix[chrom[gen], chrom[gen + 1]] 
    return weight

#Выбор и расчет длины наиболее приспособленной особи
def population_best_result(matrix, population, start_node, end_node):
    best_weight = matrix[start_node, end_node]
    best_path = []
    for chrom in range(len(population)):
        if path_weight(matrix, population[chrom]) < best_weight:
            best_weight = path_weight(matrix, population[chrom])
            best_path = population[chrom]
    return best_weight, best_path

#Скрещивание (Один  из методов эволюции популяции, при котором выбирается точка скрещивания
#и по этой точке разделяются два массива и меняются этими частями друг с другом)
def cross_one_point(population):
    population_crossed = []
    slice_line = len(population[0]) // 2
    for chrom in range(len(population)):
        X_chrom = random.choice(population)
        Y_chrom = random.choice(population)
        crossed = X_chrom[:slice_line] + Y_chrom[slice_line:]
        population_crossed.append(crossed)
        
    return population_crossed

#Мутация (Один  из методов эволюции популяции, при котором с определенным шансом 
#особь может изменить свой предложенный вариант на похожий (например поменять одну из точек в пути))
def mutation(population, count, start_node, end_node, percent_of_mutation = 0.7):
    population_mutated = []
    for chrom in range(len(population)):
        chance = random.random()
        if (chance < percent_of_mutation):
            rand_gen_ind = random.randint(1, (len(population[chrom]) - 2))
            rand_gen = random.randint(0, (count - 1))
            while (rand_gen == start_node) or (rand_gen == end_node):
                rand_gen = random.randint(0, (count - 1))
            population[chrom][rand_gen_ind] = rand_gen
            population_mutated.append(population[chrom])
        else:
            population_mutated.append(population[chrom])
            continue
    return population_mutated

#Использование библиотеки для подсчета правильного ответа кратчайшего пути
print('Кратчайший путь по алгоритму Дейкстра -', nx.dijkstra_path(graph_nx, start_node, end_node))
print('Длина кратчайшего пути по алгоритму Дейкстра -', nx.dijkstra_path_length(graph_nx, start_node, end_node))

#Поиск кратчайшего пути сравнивая особи в популяциях с уже известным ответом по алгоритму Дейкстра
gen_len = len(nx.dijkstra_path(graph_nx, start_node, end_node))
n_chrom = 10
generation = 1 
population = popul_make(start_node, end_node, gen_len, graph_nx) 

while True:
    print('Поколение: %a' % generation)
    population_selected = tournament_selection(matrix, population, n_chrom)
    population_crossed = cross_one_point(population_selected)
    population_mutated = mutation(population_crossed, count, start_node, end_node)
    population = population_mutated
    print(population)
    best_weight, best_path = population_best_result(matrix, population, start_node, end_node)
    if best_weight == nx.dijkstra_path_length(graph_nx, start_node, end_node):
        print('') 
        print('Кратчайший путь был достигнут на %a поколении' % generation)
        print('Кратчайший путь: ', nx.dijkstra_path(graph_nx, start_node, end_node))
        print('Длина кратчайшего пути: ', nx.dijkstra_path_length(graph_nx, start_node, end_node))
        
        break
    generation += 1

#Поиск кратчайшего пути не сравнивая особи в популяциях с уже известным ответом по алгоритму Дейкстра
gen_len = len(nx.dijkstra_path(graph_nx, start_node, end_node))
gen_len_end = count - 1
generation = 1  
generations = 1000
n_chrom = 10
num_first_best_gen = 1

best_weight_main = matrix[start_node, end_node]
best_path_main = []
best_generation_main = -1
best_gen_len = -1

while gen_len <= gen_len_end:
    population = popul_make(start_node, end_node, gen_len, graph_nx)
    while generation < generations:
        population_selected = tournament_selection(matrix, population, n_chrom)
        population_crossed = cross_one_point(population_selected)
        population_mutated = mutation(population_crossed, count, start_node, end_node)
        population = population_mutated
        best_weight, best_path = population_best_result(matrix, population, start_node, end_node)
        if best_weight < best_weight_main:
            best_weight_main = best_weight
            best_path_main = best_path
            generation_main = generation
            best_gen_len = gen_len
            num_first_best_gen += 1
        generation += 1
    gen_len += 1  
    
print('')    
print('Лучший результат был достигнут на %d поколении' % num_first_best_gen)
print('Кратчайший путь: ', best_path_main)
print('Длина кратчайшего пути: ', best_weight_main)
print()
print('Кратчайший путь по алгоритму Дейкстра: ', nx.dijkstra_path(graph_nx, start_node, end_node))
print('Длина кратчайшего пути по алгоритму Дейкстра: ', nx.dijkstra_path_length(graph_nx, start_node, end_node))


