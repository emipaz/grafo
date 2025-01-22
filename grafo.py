import heapq
import itertools
class Graph_Advanced():
    def __init__(self, directed=False):
        """
        Inicializar el Grafo.

        Parámetros:
        - directed (bool): Especifica si el grafo es dirigido. 
        El valor por defecto es False (no dirigido).

        Atributos:
        - graph (dict): Un diccionario para almacenar los vértices 
        y sus vértices adyacentes (con pesos).
        - directed (bool): Indica si el grafo es dirigido.
        """
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        """
        Agregar un vértice al grafo.

        Parámetros:
        - vertex: El vértice a agregar. Debe ser hasheable.

        Asegura que cada vértice esté representado en el diccionario del grafo como una clave con un diccionario vacío como su valor.
        """

        if not isinstance(vertex, (int, str, tuple)):
            raise ValueError("Vertex must be a hashable type.")
        if vertex not in self.graph:
            self.graph[vertex] = {}
    
    def add_edge(self, src, dest, weight):
        """
        Agregar una arista ponderada de src a dest. Si el grafo es no dirigido, también agregar de dest a src.

        Parámetros:
        - src: El vértice fuente.
        - dest: El vértice destino.
        - weight: El peso de la arista.
        
        Previene la adición de aristas duplicadas y asegura que ambos vértices existan.
        """
        if src not in self.graph or dest not in self.graph:
            raise KeyError("Both vertices must exist in the graph.")
        if dest not in self.graph[src]:  # Check to prevent duplicate edges
            self.graph[src][dest] = weight
        if not self.directed and src not in self.graph[dest]:
            self.graph[dest][src] = weight
    
    def remove_edge(self, src, dest):
        """
        Eliminar una arista de src a dest. Si el grafo es no dirigido, también eliminar de dest a src.

        Parámetros:
        - src: El vértice fuente.
        - dest: El vértice destino.
        """
        if src in self.graph and dest in self.graph[src]:
            del self.graph[src][dest]
        if not self.directed:
            if dest in self.graph and src in self.graph[dest]:
                del self.graph[dest][src]
    
    def remove_vertex(self, vertex):
        """
        Eliminar un vértice y todas las aristas conectadas a él.

        Parámetros:
        - vertex: El vértice que se va a eliminar.
        """
        if vertex in self.graph:
            # Remove any edges from other vertices to this one
            for adj in list(self.graph):
                if vertex in self.graph[adj]:
                    del self.graph[adj][vertex]
            # Remove the vertex entry itself
            del self.graph[vertex]
    
    def get_adjacent_vertices(self, vertex):
        """
        Obtener una lista de vértices adyacentes al vértice especificado.

        Parámetros:
        - vertex: El vértice cuyos vecinos se van a recuperar.

        Devuelve:
        - Lista de vértices adyacentes. Devuelve una lista vacía si el vértice no se encuentra.
        """
        return list(self.graph.get(vertex, {}).keys())

    def _get_edge_weight(self, src, dest):
        """
        Obtener el peso de la arista de src a dest.

        Parámetros:
        - src: El vértice fuente.
        - dest: El vértice destino.

        Devuelve:
        - El peso de la arista. Si la arista no existe, devuelve infinito.
        """
        return self.graph[src].get(dest, float('inf'))
    
    def __str__(self):
        """
        Proporcionar una representación en cadena de la lista de adyacencia del grafo para facilitar la impresión y la depuración.

        Devuelve:
        - Una representación en cadena del diccionario del grafo.
        """
        return str(self.graph)
    
    def shortest_path(self, start, end): 
        """
        Calcular el camino más corto desde un nodo de inicio hasta un nodo de destino en un grafo disperso con potencialmente miles de nodos. 
        Debe ejecutarse en menos de 0.5 segundos y encontrar la distancia más corta entre dos nodos.

        Parámetros:
        - start: El nodo de inicio.
        - end: El nodo de destino.

        Devuelve:
        Una tupla que contiene la distancia total del camino más corto y una lista de nodos que representa ese camino.
        """

        # creamos un diccionadio con las distancias en infinito
        distances = {vertex: float('inf') for vertex in self.graph}
                
        distances[start] = 0
        
        # Cola de prioridad para contener los vértices que se explorarán, inicializada con el nodo de inicio
        priority_queue = [(0, start)]
        
        # Diccionario para almacenar el camino recorrido para llegar a cada vértice
        previous_nodes = {vertex: None for vertex in self.graph}
        
        while priority_queue:
            # Obtenga el vértice con la menor distancia
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            # Si hemos llegado al nodo final, reconstruimos la ruta.
            if current_vertex == end:
                # creamos una ruta vacia
                path = []
                # mientesa el vertice no see nulo agregamos el vertice en la ruta
                while current_vertex is not None:
                    path.append(current_vertex) # agregamos el vertice actual en la ruta vamos del final al inicio
                    current_vertex = previous_nodes[current_vertex] # cambiamos el vertice_Actual y seguir el while
                # al terminar el while debolvemos la ruta invertida desde il inicio al final    
                return current_distance, path[::-1]
            
            # Si la distancia actual es mayor que la distancia más corta registrada, omitir
            if current_distance > distances[current_vertex]:
                continue
            
            # Explorar vecinos
            for neighbor, weight in self.graph[current_vertex].items():
                distance = current_distance + weight
                
                # Si se encuentra un camino más corto hacia el vecino
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_vertex # guardamos el vecino con menor distancia en previus_nodes
                    # añadir el vecino a la cola de prioridad (`priority_queue`), junto con su distancia
                    heapq.heappush(priority_queue, (distance, neighbor)) 
        
        # Si el nodo final es inalcanzable, devuelve infinito y una ruta vacía
        return float('inf'), []
        # return dist, path

    def tsp_small_graph(self, start_vertex):
        """
        Resuelve el Problema del Viajante de Comercio para un grafo completo pequeño (~10 nodos) comenzando desde un nodo especificado. 
        Se requiere encontrar el recorrido óptimo. Se esperan grafos con un máximo de 10 nodos. Debe ejecutarse en menos de 1 segundo.

        Parámetros:
        - `start_vertex`: El nodo de inicio.

        Retornos:
        - Una tupla que contiene la distancia total del recorrido y una lista de nodos que representan el camino del recorrido.
        """
        # Asegúrese de que el vértice inicial esté en el gráfo
        if start_vertex not in self.graph:
            raise KeyError("El vértice inicial debe existir en el gráfico.")
        
        # Lista de vértices excluyendo el vértice inicial
        vertices = list(self.graph.keys())
        vertices.remove(start_vertex)
        
        # Número de vértices
        n = len(vertices)
        # Mapeo de vertex por un indece
        # index_map = {vertex: i for i, vertex in enumerate(vertices)}
        index_map = {}
        for i, vertex in enumerate(vertices):
            index_map[vertex] = i
        
        # Inicializar la tabla de memorización
        memo = {}
        # Caso base: distancia desde el vértice inicial hasta cada vértice
        for i in range(n):
            memo[(1 << i, i)] = (self._get_edge_weight(start_vertex, vertices[i]), None)
            # Este bloque inicializa la tabla de memoración con las distancias desde el vértice inicial a cada vértice del subconjunto. 
            # Se utiliza un `bitmask` (`1 << i`) para representar subconjuntos de vértices.  usando << "desplazamiento de bits hacia la izq"

        # Iterar sobre subconjuntos de vértices
        for subset_size in range(2, n + 1):
            # Se itera sobre todas las combinaciones posibles de vértices de tamaño `subset_size`, 
            for subset in itertools.combinations(range(n), subset_size):
                subset_mask = sum(1 << i for i in subset)

                # Calcular la ruta más corta para cada punto final del subconjunto
                for endpoint in subset:
                    # Para cada vértice `endpoint` del subconjunto se calcula la ruta más corta desde 
                    # los vértices anteriores que ya han sido procesados. 
                    # Aquí es donde se utiliza la técnica de programación dinámica para evitar cálculos redundantes.
                    prev_mask = subset_mask & ~(1 << endpoint) # & es un AND y ~ un NOT de operadores de bits
                    min_dist = float('inf')
                    min_prev = None

                    # Encuentra la ruta más corta al punto final actual
                    for prev in subset:
                        if prev == endpoint:
                            continue
                        if (prev_mask, prev) in memo:
                            dist, _ = memo[(prev_mask, prev)]
                            dist += self._get_edge_weight(vertices[prev], vertices[endpoint])
                            
                            if dist < min_dist:
                                min_dist = dist
                                min_prev = prev
                    
                    memo[(subset_mask, endpoint)] = (min_dist, min_prev)

        # Encuentra la ruta más corta que regresa al vértice inicial
        min_distance = float('inf')
        last_vertex = None
        full_mask = (1 << n) - 1
        
        for i in range(n):
            if (full_mask, i) in memo:
                dist, _ = memo[(full_mask, i)]
                dist += self._get_edge_weight(vertices[i], start_vertex)
                
                if dist < min_distance:
                    min_distance = dist
                    last_vertex = i
        
        # Reconstruir el camino
        path = [start_vertex]
        mask = full_mask
        current_vertex = last_vertex
        while current_vertex is not None:
            # Añade el vértice actual al camino. 'vertices' es una lista o un arreglo que contiene
            # la representación (como un nombre o una posición) de los vértices.
            path.append(vertices[current_vertex])
            # Recupera el siguiente vértice en el camino a partir de la memoria (memo).
            # 'memo' es una estructura de datos que almacena resultados intermedios
            # para evitar el cálculo redundante. Se asume que contiene información sobre
            # el estado representado por 'mask' y el 'current_vertex'.
            next_vertex = memo[(mask, current_vertex)][1]
            # Actualiza la 'mask' (máscara de bits) para eliminar el 'current_vertex' de los vértices visitados.
            # '1 << current_vertex' produce un número que tiene un 1 en la posición de
            # 'current_vertex'. Al utilizar el operador NOT (~), se invierten los bits,
            # creando una máscara donde el bit en 'current_vertex' es 0 y otros son 1.
            # Luego, se realiza una operación AND (&) con 'mask' para eliminar el vértice actual de la máscara.
            mask &= ~(1 << current_vertex)
            # Actualiza 'current_vertex' para el siguiente bucle, asignándole el 
            # vértice que se obtuvo de 'memo'. Si 'next_vertex' es None, el bucle se detendrá.
            current_vertex = next_vertex
        
        path.append(start_vertex)
        
        return min_distance, path

    def tsp_large_graph(self, start):
        """
        Resolver el Problema del Viajante de Comercio para un grafo completo grande (~1000 nodos) 
        comenzando desde un nodo especificado.
        No se requiere encontrar el recorrido óptimo. Debe ejecutarse en menos de 0.5 segundos 
        con una solución "suficientemente buena".

        Parámetros:
        - start: El nodo de inicio.

        Devuelve:
        Una tupla que contiene la distancia total del recorrido y una lista de nodos que representan la ruta del recorrido.
        """
        # Asegúrese de que el vértice inicial esté en el gráfo
        if start not in self.graph:
            raise KeyError("El vértice inicial debe existir en el gráfico.")
        
        
        # iniciaizae variables
        unvisited = set(self.graph.keys())
        unvisited.remove(start)
        current_node = start
        tour = [start]
        total_distance = 0

        # Heurística del vecino más cercano
        
   
        while unvisited:
            # Mientras haya nodos no visitados, se busca el nodo más cercano 
            # desde el nodo actual (`current_node`). 
            nearest_node = None
            nearest_distance = float('inf')
            
            # Encuentra el nodo no visitado más cercano
            for node in unvisited:
                # Para cada nodo no visitado, se calcula la distancia al nodo actual 
                # utilizando el método `_get_edge_weight`. 
                distance = self._get_edge_weight(current_node, node)
                if distance < nearest_distance:
                    # Si esta distancia es menor que la distancia más cercana 
                    # encontrada hasta el momento, se actualizan `nearest_distance` y `nearest_node`.
                    nearest_distance = distance
                    nearest_node = node
            
            # Actualizamos el recorrido
            tour.append(nearest_node) # añade eñ nodo al tour
            total_distance += nearest_distance # suma la distancia
            current_node = nearest_node # movemos al siguiente nodo
            unvisited.remove(nearest_node) # eliminamos el nodo visitado
        
        # cerramos el recorrido 
        total_distance += self._get_edge_weight(current_node, start) #
        tour.append(start)
        
        return total_distance, tour