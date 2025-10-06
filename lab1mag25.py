import numpy as np
import matplotlib.pyplot as plt

#Распределение Пуассона 
N = 100000
lamda = 150
data = np.random.poisson(lamda, size=N)
#print("Исходный массив:", data)

#Экспоненциальное распределение 
#N = 100000
#lamda = 150
#data = np.random.poisson(lamda, size=N)
#print("Исходный массив:", data)

def genetic_algorithm(data, population_size, selection_size, generations, P_crossover, P_mutation):
    population = np.random.randint(0, len(data), size=population_size)
    value_history = [] 
    
    for generation in range(generations):        
        parents = []
        for _ in range(population_size):
            candidates = np.random.choice(population, selection_size)
            candidate_values = data[candidates]  
            best_index = np.argmax(candidate_values) 
            best_cand = candidates[best_index]  
            parents.append(best_cand)
        
        offspring = []
        for i in range(0, len(parents), 2):  
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i+1]               
                if np.random.random() < P_crossover:
                    alpha = np.random.random()  
                    child1 = int(alpha * parent1 + (1 - alpha) * parent2)
                    child2 = int(alpha * parent2 + (1 - alpha) * parent1)
                else:
                    child1, child2 = parent1, parent2
                    
                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])
        
        population = np.array(offspring)  
        
        for i in range(len(population)):
            if np.random.random() < P_mutation:
                population[i] = np.random.randint(0, len(data))
        
        best_idx = population[np.argmax(data[population])]
        best_value = data[best_idx]
        value_history.append(best_value) 
        
        #print(f"Поколение {generation}: Максимум={best_value}")
    
    best_idx = population[np.argmax(data[population])]
    return data[best_idx], value_history 

result_max, value_history = genetic_algorithm(
    data, population_size=500, selection_size = 15, generations=100, P_crossover =0.5, P_mutation=0.15)
real_max = np.max(data) 

print(f"\nПолученный максимум: {result_max}")
print(f"Реальный максимум: {real_max}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(value_history)), value_history, 'b-', linewidth=2)
plt.axhline(y=real_max, color='r', linestyle='--', linewidth=2, label='Реальный максимум')
plt.xlabel('Поколение')
plt.ylabel('Лучшее значение')
plt.title('Сходимость генетического алгоритма')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
