import heapq

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