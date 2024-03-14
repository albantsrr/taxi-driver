import random

GLB_NB_ROW = 7
GLB_NB_COLUMN = 12 
map_matrice = [['F' for _ in range(GLB_NB_COLUMN)] for _ in range(GLB_NB_ROW)]


number_of_state = GLB_NB_ROW*GLB_NB_COLUMN 
nb_hole = int(number_of_state*0.2)


list_indice_hole = []
indice_goal = None

list_indice_hole = random.sample(range(number_of_state), nb_hole)
indice_goal = random.randint(4, number_of_state-1)
while(indice_goal in list_indice_hole):
    indice_goal = random.randint(4, number_of_state-1)

for i in range(len(list_indice_hole)):
    indice = list_indice_hole[i]
    row_indice = indice // GLB_NB_COLUMN
    column_indice = indice % GLB_NB_ROW

    map_matrice[row_indice][column_indice] = 'H'


row_indice = indice_goal // GLB_NB_COLUMN
column_indice = indice_goal % GLB_NB_ROW
map_matrice[row_indice][column_indice] = 'G'


custom_map = [''.join(row) for row in map_matrice]
print(custom_map)