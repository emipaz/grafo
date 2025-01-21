

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