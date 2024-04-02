from dataclasses import dataclass
from typing import List, Dict
from copy import copy

@dataclass
class Line:
    a: float
    b: float
    border_left: float
    border_right: float

    def __sub__(self, other):
        if (self.border_left != other.border_left) or (self.border_right != other.border_right):
            raise Exception('Different borders')
        return Line(self.a - other.a, self.b - other.b,self.border_left, self.border_right ) 


    def zero_split(self):
        if self.a == 0:
            return [self] 
        x_intersect = -self.b / self.a
        if (x_intersect < self.border_right) and (x_intersect > self.border_left):
            line_l, line_r = copy(self), copy(self)
            line_l.border_right = x_intersect
            line_r.border_left = x_intersect
            return [line_l, line_r]
        else:
            return [self]

    def integrate(self) -> float:
        alpha = self.border_left
        beta = self.border_right
        return abs(self.a/2 * (beta ** 2 - alpha**2) + self.b * (beta - alpha))
       
N, M = input().split()
N = int(N)
M = int(M)
x_f = [ float(i) for i in input().split()]
a_b_f = []
for _ in range(N):
    a, b = input().split()
    a, b = int(a), int(b)
    a_b_f.append((a, b))
x_g = [ float(i) for i in input().split()]
a_b_g = []
for _ in range(M):
    a, b = input().split()
    a, b = int(a), int(b)
    a_b_g.append((a, b))


f_lines = []
for i in range(len(x_f) - 1):
    f_lines.append(Line(a_b_f[i][0], a_b_f[i][1], x_f[i], x_f[i+1]))

g_lines = []
for i in range(len(x_g) - 1):
    g_lines.append(Line(a_b_g[i][0], a_b_g[i][1], x_g[i], x_g[i+1]))


def intervals_parser(x_f, x_g) -> List[float]:
    'Обьединить массивы, отсориторвать и дропнуть дубликаты'
    a = x_f + x_g
    a = list(set(a))
    a.sort()

    return a


def params_parser(lines_list : List[Line], point, index) -> List[Line]:
    'Для каждого отрезка смотрим какие параметры попадают'
    # tmp_lines = copy(lines_list)
    while True: 
    # for i, line in enumerate(tmp_lines):
        line = lines_list[index]
        if (point > line.border_left) and (point < line.border_right):
            return line, index
        if (point > line.border_right):
            index += 1
    
    
points = intervals_parser(x_f, x_g)

integral = 0
f_index = 0
g_index = 0
for i in range(len(points) - 1):
    segment_point = points[i+1] + points[i]
    segment_point *= 0.5
    
    line_f, f_index = params_parser(f_lines, segment_point, f_index)
    line_g, g_index = params_parser(g_lines, segment_point, g_index)
    line = Line(line_f.a-line_g.a, line_f.b - line_g.b, points[i], points[i+1])
    for subline in line.zero_split():
        integral += subline.integrate()

print(f'{integral:.4f}')