#!/usr/bin/env python3
"""SVG path 数据转 aardio gdip.path 代码的转换工具"""
import re
import json
import math

def parse_svg_path(d):
    """解析SVG path的d属性，返回命令列表"""
    # 将命令字母和数字分离
    tokens = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
    commands = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            cmd = token
            i += 1
            params = []
            while i < len(tokens) and not tokens[i].isalpha():
                params.append(float(tokens[i]))
                i += 1
            commands.append((cmd, params))
        else:
            i += 1
    return commands

def q_to_c(cx, cy, qx, qy, ex, ey):
    """二次贝塞尔转三次贝塞尔"""
    cp1x = cx + 2/3 * (qx - cx)
    cp1y = cy + 2/3 * (qy - cy)
    cp2x = ex + 2/3 * (qx - ex)
    cp2y = ey + 2/3 * (qy - ey)
    return cp1x, cp1y, cp2x, cp2y, ex, ey

def svg_to_aardio(d, target_size=40):
    """将SVG path d属性转为aardio gdip.path代码"""
    commands = parse_svg_path(d)
    if not commands:
        return None
    
    # 第一遍：计算所有点的坐标和bounding box
    cx, cy = 0, 0  # 当前点
    sx, sy = 0, 0  # 子路径起点
    all_points = []
    last_cmd = ''
    last_cp = None  # 上一个控制点（用于S/T）
    
    # 收集所有坐标点用于计算bounding box
    temp_segments = []
    
    for cmd, params in commands:
        if cmd in ('M', 'm'):
            if cmd == 'M':
                cx, cy = params[0], params[1]
            else:
                cx, cy = cx + params[0], cy + params[1]
            sx, sy = cx, cy
            all_points.append((cx, cy))
            last_cp = None
            # 处理多余的点对作为L
            for j in range(2, len(params), 2):
                if j+1 < len(params):
                    if cmd == 'M':
                        cx, cy = params[j], params[j+1]
                    else:
                        cx, cy = cx + params[j], cy + params[j+1]
                    all_points.append((cx, cy))
                    
        elif cmd in ('L', 'l'):
            for j in range(0, len(params), 2):
                if cmd == 'L':
                    nx, ny = params[j], params[j+1]
                else:
                    nx, ny = cx + params[j], cy + params[j+1]
                all_points.append((nx, ny))
                cx, cy = nx, ny
            last_cp = None
            
        elif cmd in ('H', 'h'):
            for p in params:
                if cmd == 'H':
                    nx = p
                else:
                    nx = cx + p
                ny = cy
                all_points.append((nx, ny))
                cx, cy = nx, ny
            last_cp = None
            
        elif cmd in ('V', 'v'):
            for p in params:
                nx = cx
                if cmd == 'V':
                    ny = p
                else:
                    ny = cy + p
                all_points.append((nx, ny))
                cx, cy = nx, ny
            last_cp = None
            
        elif cmd in ('C', 'c'):
            for j in range(0, len(params), 6):
                if j+5 < len(params):
                    if cmd == 'C':
                        cp1x, cp1y = params[j], params[j+1]
                        cp2x, cp2y = params[j+2], params[j+3]
                        ex, ey = params[j+4], params[j+5]
                    else:
                        cp1x, cp1y = cx + params[j], cy + params[j+1]
                        cp2x, cp2y = cx + params[j+2], cy + params[j+3]
                        ex, ey = cx + params[j+4], cy + params[j+5]
                    all_points.extend([(cp1x, cp1y), (cp2x, cp2y), (ex, ey)])
                    last_cp = (cp2x, cp2y)
                    cx, cy = ex, ey
                    
        elif cmd in ('S', 's'):
            for j in range(0, len(params), 4):
                if j+3 < len(params):
                    if last_cp:
                        cp1x, cp1y = 2*cx - last_cp[0], 2*cy - last_cp[1]
                    else:
                        cp1x, cp1y = cx, cy
                    if cmd == 'S':
                        cp2x, cp2y = params[j], params[j+1]
                        ex, ey = params[j+2], params[j+3]
                    else:
                        cp2x, cp2y = cx + params[j], cy + params[j+1]
                        ex, ey = cx + params[j+2], cy + params[j+3]
                    all_points.extend([(cp1x, cp1y), (cp2x, cp2y), (ex, ey)])
                    last_cp = (cp2x, cp2y)
                    cx, cy = ex, ey
                    
        elif cmd in ('Q', 'q'):
            for j in range(0, len(params), 4):
                if j+3 < len(params):
                    if cmd == 'Q':
                        qx, qy = params[j], params[j+1]
                        ex, ey = params[j+2], params[j+3]
                    else:
                        qx, qy = cx + params[j], cy + params[j+1]
                        ex, ey = cx + params[j+2], cy + params[j+3]
                    all_points.extend([(qx, qy), (ex, ey)])
                    last_cp = (qx, qy)
                    cx, cy = ex, ey
                    
        elif cmd in ('T', 't'):
            for j in range(0, len(params), 2):
                if j+1 < len(params):
                    if last_cp:
                        qx, qy = 2*cx - last_cp[0], 2*cy - last_cp[1]
                    else:
                        qx, qy = cx, cy
                    if cmd == 'T':
                        ex, ey = params[j], params[j+1]
                    else:
                        ex, ey = cx + params[j], cy + params[j+1]
                    all_points.extend([(qx, qy), (ex, ey)])
                    last_cp = (qx, qy)
                    cx, cy = ex, ey
                    
        elif cmd in ('Z', 'z'):
            cx, cy = sx, sy
            last_cp = None
            
        last_cmd = cmd
    
    if not all_points:
        return None
    
    # 计算bounding box
    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_x = max(p[0] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    bw = max_x - min_x
    bh = max_y - min_y
    if bw == 0: bw = 1
    if bh == 0: bh = 1
    
    # 计算缩放和偏移，使图标居中到(0,0)，大小为target_size
    scale = min(target_size / bw, target_size / bh)
    offset_x = -(min_x + bw/2) * scale
    offset_y = -(min_y + bh/2) * scale
    
    def tx(x):
        return round(x * scale + offset_x, 2)
    def ty(y):
        return round(y * scale + offset_y, 2)
    
    # 第二遍：生成aardio代码
    lines = []
    cx, cy = 0, 0
    sx, sy = 0, 0
    last_cp = None
    figure_started = False
    
    for cmd, params in commands:
        if cmd in ('M', 'm'):
            if figure_started:
                lines.append('\t\tpath.closeFigure();')
            if cmd == 'M':
                cx, cy = params[0], params[1]
            else:
                cx, cy = cx + params[0], cy + params[1]
            sx, sy = cx, cy
            lines.append(f'\t\tpath.startFigure();')
            figure_started = True
            last_cp = None
            for j in range(2, len(params), 2):
                if j+1 < len(params):
                    ox, oy = tx(cx), ty(cy)
                    if cmd == 'M':
                        cx, cy = params[j], params[j+1]
                    else:
                        cx, cy = cx + params[j], cy + params[j+1]
                    nx, ny = tx(cx), ty(cy)
                    lines.append(f'\t\tpath.addLine({ox},{oy},{nx},{ny});')
                    
        elif cmd in ('L', 'l'):
            for j in range(0, len(params), 2):
                ox, oy = tx(cx), ty(cy)
                if cmd == 'L':
                    cx, cy = params[j], params[j+1]
                else:
                    cx, cy = cx + params[j], cy + params[j+1]
                nx, ny = tx(cx), ty(cy)
                lines.append(f'\t\tpath.addLine({ox},{oy},{nx},{ny});')
            last_cp = None
            
        elif cmd in ('H', 'h'):
            for p in params:
                ox, oy = tx(cx), ty(cy)
                if cmd == 'H':
                    cx = p
                else:
                    cx = cx + p
                nx, ny = tx(cx), ty(cy)
                lines.append(f'\t\tpath.addLine({ox},{oy},{nx},{ny});')
            last_cp = None
            
        elif cmd in ('V', 'v'):
            for p in params:
                ox, oy = tx(cx), ty(cy)
                if cmd == 'V':
                    cy = p
                else:
                    cy = cy + p
                nx, ny = tx(cx), ty(cy)
                lines.append(f'\t\tpath.addLine({ox},{oy},{nx},{ny});')
            last_cp = None
            
        elif cmd in ('C', 'c'):
            for j in range(0, len(params), 6):
                if j+5 < len(params):
                    if cmd == 'C':
                        cp1x, cp1y = params[j], params[j+1]
                        cp2x, cp2y = params[j+2], params[j+3]
                        ex, ey = params[j+4], params[j+5]
                    else:
                        cp1x, cp1y = cx + params[j], cy + params[j+1]
                        cp2x, cp2y = cx + params[j+2], cy + params[j+3]
                        ex, ey = cx + params[j+4], cy + params[j+5]
                    x0, y0 = tx(cx), ty(cy)
                    x1, y1 = tx(cp1x), ty(cp1y)
                    x2, y2 = tx(cp2x), ty(cp2y)
                    x3, y3 = tx(ex), ty(ey)
                    lines.append(f'\t\tpath.addBezier({x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3});')
                    last_cp = (cp2x, cp2y)
                    cx, cy = ex, ey
                    
        elif cmd in ('S', 's'):
            for j in range(0, len(params), 4):
                if j+3 < len(params):
                    if last_cp:
                        cp1x, cp1y = 2*cx - last_cp[0], 2*cy - last_cp[1]
                    else:
                        cp1x, cp1y = cx, cy
                    if cmd == 'S':
                        cp2x, cp2y = params[j], params[j+1]
                        ex, ey = params[j+2], params[j+3]
                    else:
                        cp2x, cp2y = cx + params[j], cy + params[j+1]
                        ex, ey = cx + params[j+2], cy + params[j+3]
                    x0, y0 = tx(cx), ty(cy)
                    x1, y1 = tx(cp1x), ty(cp1y)
                    x2, y2 = tx(cp2x), ty(cp2y)
                    x3, y3 = tx(ex), ty(ey)
                    lines.append(f'\t\tpath.addBezier({x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3});')
                    last_cp = (cp2x, cp2y)
                    cx, cy = ex, ey
                    
        elif cmd in ('Q', 'q'):
            for j in range(0, len(params), 4):
                if j+3 < len(params):
                    if cmd == 'Q':
                        qx, qy = params[j], params[j+1]
                        ex, ey = params[j+2], params[j+3]
                    else:
                        qx, qy = cx + params[j], cy + params[j+1]
                        ex, ey = cx + params[j+2], cy + params[j+3]
                    # 转为三次贝塞尔
                    cp1x, cp1y, cp2x, cp2y, eex, eey = q_to_c(cx, cy, qx, qy, ex, ey)
                    x0, y0 = tx(cx), ty(cy)
                    x1, y1 = tx(cp1x), ty(cp1y)
                    x2, y2 = tx(cp2x), ty(cp2y)
                    x3, y3 = tx(eex), ty(eey)
                    lines.append(f'\t\tpath.addBezier({x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3});')
                    last_cp = (qx, qy)
                    cx, cy = ex, ey
                    
        elif cmd in ('T', 't'):
            for j in range(0, len(params), 2):
                if j+1 < len(params):
                    if last_cp:
                        qx, qy = 2*cx - last_cp[0], 2*cy - last_cp[1]
                    else:
                        qx, qy = cx, cy
                    if cmd == 'T':
                        ex, ey = params[j], params[j+1]
                    else:
                        ex, ey = cx + params[j], cy + params[j+1]
                    cp1x, cp1y, cp2x, cp2y, eex, eey = q_to_c(cx, cy, qx, qy, ex, ey)
                    x0, y0 = tx(cx), ty(cy)
                    x1, y1 = tx(cp1x), ty(cp1y)
                    x2, y2 = tx(cp2x), ty(cp2y)
                    x3, y3 = tx(eex), ty(eey)
                    lines.append(f'\t\tpath.addBezier({x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3});')
                    last_cp = (qx, qy)
                    cx, cy = ex, ey
                    
        elif cmd in ('Z', 'z'):
            if figure_started:
                lines.append('\t\tpath.closeFigure();')
                figure_started = False
            cx, cy = sx, sy
            last_cp = None
    
    if figure_started:
        lines.append('\t\tpath.closeFigure();')
    
    if not lines:
        return None
    
    return '\n'.join(lines)


def main():
    # 读取JS文件中的图标数据
    with open(r'c:\Projects\aardio\Keybored\cute-svg-icons.js', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有path数据
    icons = []
    pattern = r'\{\s*name:\s*"([^"]+)"\s*,\s*path:\s*"([^"]+)"\s*\}'
    for match in re.finditer(pattern, content):
        name = match.group(1)
        path_d = match.group(2)
        icons.append((name, path_d))
    
    print(f"找到 {len(icons)} 个图标")
    
    # 转换并生成aardio代码
    builders = []
    failed = []
    
    for idx, (name, path_d) in enumerate(icons):
        aardio_code = svg_to_aardio(path_d)
        if aardio_code:
            builders.append((name, aardio_code))
        else:
            failed.append(name)
    
    print(f"成功转换 {len(builders)} 个，失败 {len(failed)} 个")
    if failed:
        print(f"失败列表: {failed[:10]}...")
    
    # 生成shapeData.aardio
    lines = []
    lines.append('//shapeData 矢量图形数据模块')
    lines.append('//由SVG path数据自动转换生成，所有图形归一化到40x40像素，中心(0,0)')
    lines.append('import gdip.path;')
    lines.append('')
    lines.append('namespace shapeData;')
    lines.append('')
    
    for idx, (name, code) in enumerate(builders):
        func_name = f'build{idx+1}'
        lines.append(f'//{name}')
        lines.append(f'{func_name} = function(){{')
        lines.append('\tvar path = ..gdip.path(1/*_FillModeWinding*/);')
        lines.append(code)
        lines.append('\treturn path;')
        lines.append('}')
        lines.append('')
    
    # 生成构建器列表
    lines.append('//图形构建器列表')
    lines.append('var builders = {')
    for idx in range(len(builders)):
        comma = ',' if idx < len(builders) - 1 else ''
        lines.append(f'\tbuild{idx+1}{comma}')
    lines.append('};')
    lines.append('')
    
    # 生成获取所有path的方法
    lines.append('//批量构建所有图形路径')
    lines.append('buildAll = function(){')
    lines.append('\tvar paths = {};')
    lines.append('\tfor(i=1;#builders;1){')
    lines.append('\t\t..table.push(paths, builders[i]());')
    lines.append('\t}')
    lines.append('\treturn paths;')
    lines.append('}')
    lines.append('')
    
    # 生成intellisense
    lines.append('/**intellisense()')
    lines.append('shapeData.buildAll() = 批量构建所有图形路径，返回path对象数组')
    for idx, (name, _) in enumerate(builders):
        lines.append(f'shapeData.build{idx+1}() = {name}')
    lines.append('end intellisense**/')
    
    output = '\n'.join(lines)
    
    with open(r'c:\Projects\aardio\Keybored\lib\shapeData.aardio', 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"已生成 shapeData.aardio，共 {len(builders)} 个图形构建函数")


if __name__ == '__main__':
    main()
