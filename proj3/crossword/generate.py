import sys

from crossword import *

#Additional imports 
from queue import Queue


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            domain = self.domains[v].copy()
            for x in domain:
                if v.length != len(x):
                    self.domains[v].remove(x)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision = False

        x_domain = self.domains[x].copy()
        for x_value in x_domain:
            arc_check = False
            for y_value in self.domains[y]:
                i, j = self.crossword.overlaps[x, y]
                if x_value[i] == y_value[j]:
                    arc_check = True

            if not arc_check:
                self.domains[x].remove(x_value)
                revision = True

        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        q = Queue()

        if arcs is None:
            #Creating a queue of all arcs in the problem if arcs is None
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    q.put((x, y))
        else:
            for pair in arcs:
                q.put(pair)

        while not q.empty():
            x, y = q.get()

            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                #Adding neighbors of x, to ensure that they stay consistent with x
                for z in self.crossword.neighbors(x) - {y}:
                    q.put((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(self.crossword.variables) == len(assignment):
            return True
        else:
            return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        #Checking whether all values are distinct
        for x in assignment:
            for y in assignment:
                if x != y and assignment[x] == assignment[y]:
                    return False 

        #Checking if every value is the correct length
        for var in assignment:
            if var.length != len(assignment[var]):
                return False

        #Ensuring there are no conflicts between neighboring variables
        for x in assignment:
            for y in self.crossword.neighbors(x):
                if y in assignment:
                    i, j = self.crossword.overlaps[x, y]
                    if assignment[x][i] != assignment[y][j]:
                        return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        return sorted(self.domains[var], key=lambda value: self.least_constraining_values(var, value, assignment))

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_vars = []

        for var in self.crossword.variables:
            if var not in assignment:
                unassigned_vars.append(var)

        #Sorting unassigned variables according to their domain size in ascending order
        mrv = sorted(unassigned_vars, key=lambda v: len(self.domains[v]))

        degree = []
        min_domain = len(self.domains[mrv[0]])
        for var in mrv:
            if len(self.domains[var]) == min_domain:
                degree.append(var)

        #Ensuring there is no tie for the variable with the fewest number of remaining values
        if len(degree) == 1:
            return degree[0]
        else:
            #Since there is a tie, sorting according to their degree in descending order
            degree.sort(key=lambda v: len(self.crossword.neighbors(v)), reverse=True)
            return degree[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None

    def least_constraining_values(self, var, value, assignment):
        """
        Returns the number of values ruled out for neighboring unassigned variables,
        when assigning the variable 'var' with a value of 'value'
        """
        n = 0

        for neighbor in self.crossword.neighbors(var):
            if neighbor not in assignment:
                for neighbor_value in self.domains[neighbor]:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if value[i] != neighbor_value[j]:
                        n += 1

        return n


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
