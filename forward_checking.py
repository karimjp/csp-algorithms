import networkx as nx
import numpy as np
import pdb
import copy
little_europe = {
    "nodes":  ["Iceland", "Ireland", "United Kingdom", "Portugal", "Spain",
                 "France", "Belgium", "Netherlands", "Luxembourg", "Germany"
                  ], 
    "edges": [(0,1), (0,2), (1,2), (2,5), (2,6), (2,7), (2,11), (3,4),
                 (4,5), (4,22), (5,6), (5,8), (5,9), (5,21), (5,22),(6,7),
                 (6,8), (6,9), (7,9), (8,9), (9,10), (9,12), (9,17), (9,18),
                 (9,19), (9,21)],
    "coordinates": [( 18,147), ( 48, 83), ( 64, 90), ( 47, 28), ( 63, 34),
                   ( 78, 55), ( 82, 74), ( 84, 80), ( 82, 69), (100, 78),
                   ( 94, 97), (110,162), (116,144), (143,149), (140,111),
                   (137,102), (136, 95), (122, 78), (110, 67), (112, 60),
                   ( 98, 59), ( 93, 55), (102, 35), (108, 14), (130, 22),
                   (125, 32), (128, 37), (127, 40), (122, 42), (118, 47),
                   (127, 48), (116, 53), (111, 54), (122, 57), (124, 65),
                   (146, 87), (158, 65), (148, 57), (138, 54), (137, 41),
                   (160, 13), (168, 29), (189, 39), (194, 32), (202, 33),
                   (191,118)]}
europe = {
    "nodes":  ["Iceland", "Ireland", "United Kingdom", "Portugal", "Spain",
                 "France", "Belgium", "Netherlands", "Luxembourg", "Germany",
                 "Denmark", "Norway", "Sweden", "Finland", "Estonia",
                 "Latvia", "Lithuania", "Poland", "Czech Republic", "Austria",
                 "Liechtenstein", "Switzerland", "Italy", "Malta", "Greece",
                 "Albania", "Macedonia", "Kosovo", "Montenegro", "Bosnia Herzegovina",
                 "Serbia", "Croatia", "Slovenia", "Hungary", "Slovakia",
                 "Belarus", "Ukraine", "Moldova", "Romania", "Bulgaria",
                 "Cyprus", "Turkey", "Georgia", "Armenia", "Azerbaijan",
                 "Russia" ], 
    "edges": [(0,1), (0,2), (1,2), (2,5), (2,6), (2,7), (2,11), (3,4),
                 (4,5), (4,22), (5,6), (5,8), (5,9), (5,21), (5,22),(6,7),
                 (6,8), (6,9), (7,9), (8,9), (9,10), (9,12), (9,17), (9,18),
                 (9,19), (9,21), (10,11), (10,12), (10,17), (11,12), (11,13), (11,45), 
                 (12,13), (12,14), (12,15), (12,17), (13,14), (13,45), (14,15),
                 (14,45), (15,16), (15,35), (15,45), (16,17), (16,35), (17,18),
                 (17,34), (17,35), (17,36), (18,19), (18,34), (19,20), (19,21), 
                 (19,22), (19,32), (19,33), (19,34), (20,21), (21,22), (22,23),
                 (22,24), (22,25), (22,28), (22,29), (22,31), (22,32), (24,25),
                 (24,26), (24,39), (24,40), (24,41), (25,26), (25,27), (25,28),
                 (26,27), (26,30), (26,39), (27,28), (27,30), (28,29), (28,30),
                 (29,30), (29,31), (30,31), (30,33), (30,38), (30,39), (31,32),
                 (31,33), (32,33), (33,34), (33,36), (33,38), (34,36), (35,36),
                 (35,45), (36,37), (36,38), (36,45), (37,38), (38,39), (39,41),
                 (40,41), (41,42), (41,43), (41,44), (42,43), (42,44), (42,45),
                 (43,44), (44,45)],
    "coordinates": [( 18,147), ( 48, 83), ( 64, 90), ( 47, 28), ( 63, 34),
                   ( 78, 55), ( 82, 74), ( 84, 80), ( 82, 69), (100, 78),
                   ( 94, 97), (110,162), (116,144), (143,149), (140,111),
                   (137,102), (136, 95), (122, 78), (110, 67), (112, 60),
                   ( 98, 59), ( 93, 55), (102, 35), (108, 14), (130, 22),
                   (125, 32), (128, 37), (127, 40), (122, 42), (118, 47),
                   (127, 48), (116, 53), (111, 54), (122, 57), (124, 65),
                   (146, 87), (158, 65), (148, 57), (138, 54), (137, 41),
                   (160, 13), (168, 29), (189, 39), (194, 32), (202, 33),
                   (191,118)]}

connecticut = { "nodes": ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlesex", "Tolland", "New London", "Windham"],
                "edges": [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6), (4,6), (5,6), (5,7), (6,7)],
                "coordinates": [( 46, 52), ( 65,142), (104, 77), (123,142), (147, 85), (162,140), (197, 94), (217,146)]}
#inference affect variable connections constraints
#order domain values affect variable constraints
#select unassigned variables affect unassigned stack 
def unassigned_list_setup(planar_map, colors):
    #get child neighbors and parent neighbors
    def neighbor_index_list_generator(node_index, edges):
        neighbors = []
        for vertex1,vertex2 in edges:
            if vertex1 == node_index:
                neighbors.append(vertex2)
            elif vertex2 == node_index:
                neighbors.append(vertex1)
        return neighbors
    
    #(variable_index, constraints, connections, assignment)
    null_assignment = -1 
    unassigned_list = [[node_index,[index for index, color in enumerate(colors)]
                        , neighbor_index_list_generator(node_index, planar_map["edges"]), null_assignment]
                            for node_index,node in enumerate(planar_map["nodes"])]
    for variable in unassigned_list:
        variable[2].append(variable[0])
    return  unassigned_list

def select_unassigned_variable(unassigned_list):
    #returns first node in ordered list
    return unassigned_list[0]

def remove_assigned_variable(selected_variable, unassigned_list, assigned_list):
    assigned_list.insert(0,unassigned_list.pop(0))

def find_interconnections(neighbors):
    neighbors_interconnection ={}
    for pivot_neighbor in neighbors:
        pivot_index = pivot_neighbor[0]
        pivot_connections = pivot_neighbor[1]
        neighbors_interconnection[pivot_index] = []
        for neighbor in neighbors:
            neighbor_index = neighbor[0]
            if neighbor_index in pivot_connections:
                neighbors_interconnection[pivot_index].append(neighbor_index)
    #print neighbors_interconnection
    return neighbors_interconnection


def is_cycle_redundant(neighbors_interconnection):
    for pivot_index, pivot_connection in neighbors_interconnection.iteritems():
        redundant_neighbors = []
        if len(pivot_connection) % 2: #verifies odd number of regions
            for neighbor_index, connection in neighbors_interconnection.iteritems():
                if set(pivot_connection).issubset(connection):
                    redundant_neighbors.append(neighbor_index)
        if redundant_neighbors.sort() == pivot_connection.sort():
            return True
    return False
                    
                        
def four_color_theorem_invalid_test(selected_variable, unassigned_list, assigned_list):
    #A map in which one region is surrounded by an odd number of other regions
    #that touch each other in a cycle.
    neighbors = neighbor_list_generator(selected_variable, unassigned_list, assigned_list)
    neighbors = [(variable_index, connections) for variable_index, constraints, connections, assignment in neighbors 
                 if selected_variable[0] != variable_index]
    neighbors_interconnection = find_interconnections(neighbors)
    if is_cycle_redundant(neighbors_interconnection):
        return True
    return False
    
            

def order_domain_values(selected_variable, unassigned_list):
    #for backtrack this is left in order for static order selection
    values = selected_variable[1]
    return values


def neighbor_list_generator(selected_variable, unassigned_list, assigned_list=[]):
    neighbors = []
    #pdb.set_trace()
    selected_variable_index = selected_variable[0]
    #print "selected: ", selected_variable_index
    check_lists = [unassigned_list, assigned_list]
    #print "check list sizes: ", len(unassigned_list), len(assigned_list)
    for check_list in check_lists:
        for variable in check_list:
            #print variable
            variable_connections = variable[2]
            if selected_variable_index in variable_connections:
                neighbors.append(variable)
    return neighbors


def is_consistent_value(value, selected_variable, unassigned_list, assigned_list):
    neighbors = neighbor_list_generator(selected_variable, unassigned_list, assigned_list)
    for neighbor in neighbors:
        variable_assignment = neighbor[3]
        if value == variable_assignment:
            #print "inconsistent adjacent"
            return False
    #print "consistent adjacent"
    return True

def assign_value(selected_variable, value):
    assignment_index = 3     
    selected_variable[assignment_index] = value

def solution(assigned_list, planar_map, colors):
    assigned_list.reverse()
    assigned_list = [(planar_map["nodes"][variable[0]], colors[variable[3]]) for variable in assigned_list]
    return assigned_list

def make_arc_consistent(selected_variable, unassigned_list):
    pre_changed_variables = []
    checklists = [unassigned_list]
    constraints_index = 1
    selected_variable_assignment = selected_variable[3]
    neighbors = neighbor_list_generator(selected_variable, unassigned_list)
    if selected_variable_assignment in [assignment for variable_index, constraints, 
                                     connections, assignment in neighbors]:
        for checklist in checklists:
            for variable in checklist:
                if variable in neighbors:
                    if selected_variable_assignment in variable[constraints_index]:
                        pre_changed_variables.append(copy.deepcopy(variable))
                        variable[constraints_index].remove(selected_variable_assignment)
                        print variable
    return pre_changed_variables

def restore(unassigned_list, pre_changed_variables):
    print "RESTORING: ", pre_changed_variables
    checklists=[unassigned_list]
    for variable in pre_changed_variables:
        for checklist in checklists:
            var_indexes = [check_var[0] for check_var in checklist]
            if variable[0] in var_indexes:
                index = var_indexes.index(variable[0])
                print index
                checklist[index] = variable
                print checklist
            
def revise(neighbors):
    constraints_index = 1
    assignment_index = 3
    for neighbor in neighbors:
        if len(neighbor[constraints_index]) == 0 and neighbor[assignment_index] == -1:
            return False
    return True
        
"""def forward_checking(selected_variable, unassigned_list, assigned_list):
    pre_changed_variables  = make_arc_consistent(selected_variable, unassigned_list, assigned_list)
    neighbors = neighbor_list_generator(selected_variable, unassigned_list, assigned_list)
    if not revise(neighbors):
        print "EMPTY NEIGHBORS!!!"
        restore(unassigned_list, assigned_list, pre_changed_variables)"""
        
def forward_checking(selected_variable, unassigned_list):
    pre_changed_variables  = make_arc_consistent(selected_variable, unassigned_list)
    neighbors = neighbor_list_generator(selected_variable, unassigned_list)
    if not revise(neighbors):
        print "EMPTY NEIGHBORS!!!"
        restore(unassigned_list, pre_changed_variables)
def backtrack(selected_variable, unassigned_list, assigned_list, colors):
    print "backtracking"
    print selected_variable
    null_assignment = -1
    assign_value(selected_variable, null_assignment)
    unassigned_list.insert(0,assigned_list.pop(0))
    if len(assigned_list) == 0:
        return False
    if len(colors) < 4:
        if four_color_theorem_invalid_test(selected_variable, unassigned_list, assigned_list):
            return False
    return True

def inference(selected_variable, unassigned_list):
    #for backtrack this is not used unless we applies an inference algorithm like forward checking
    forward_checking(selected_variable, unassigned_list)


def backtrack_search(planar_map, colors, trace):
    assigned_list = []
    unassigned_list = unassigned_list_setup(planar_map, colors)
    #print unassigned_list
    while len(unassigned_list) != 0:
        attempts = 0
        selected_variable = select_unassigned_variable(unassigned_list)
        #pdb.set_trace()
        if trace:
            print "______________________"
            print "Selected: ", planar_map["nodes"][selected_variable[0]]
        
        for value in order_domain_values(selected_variable, unassigned_list):
            #print "testing value:", value
            attempts = attempts + 1
            values_length = len(selected_variable[1])
            if is_consistent_value(value, selected_variable, unassigned_list, assigned_list):
                #print "color: ", colors[value]
                assign_value(selected_variable, value)
                inference(selected_variable, unassigned_list)
                remove_assigned_variable(selected_variable, unassigned_list, assigned_list)
                
                break
            #print "inconsistent value ", value
                
            if attempts == values_length:
                #print "attempts:",attempts,"values: ", values_length
                if not backtrack(selected_variable, unassigned_list, assigned_list, colors):
                    return None
    return solution(assigned_list, planar_map, colors)
            
                        
def color_map(planar_map, colors, trace=False):
    print backtrack_search(planar_map,colors, trace)

color_map(connecticut, ["red", "blue", "green"], trace=True)


