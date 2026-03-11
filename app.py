import streamlit as st
import cv2
import numpy as np
from collections import deque
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import heapq


# --------------------------------------
# Center-Preferred Path using Dijkstra
# --------------------------------------
def centered_path(grid, start, end):

    h, w = grid.shape

    # Distance transform gives corridor center strength
    dist_map = cv2.distanceTransform(grid, cv2.DIST_L2, 5)

    pq = []
    heapq.heappush(pq, (0, start))
    parent = {start: None}
    cost = {start: 0}

    while pq:
        cur_cost, (r, c) = heapq.heappop(pq)

        if (r, c) == end:
            break

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr, nc] == 255:

                    # Prefer pixels farthest from walls
                    penalty = 1 / (dist_map[nr, nc] + 1)

                    new_cost = cur_cost + penalty

                    if (nr, nc) not in cost or new_cost < cost[(nr, nc)]:
                        cost[(nr, nc)] = new_cost
                        parent[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (new_cost, (nr, nc)))

    return parent


# --------------------------------------
# Reconstruct Path
# --------------------------------------
def reconstruct(parent, start, end):
    path = []
    cur = end

    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)

    path.reverse()

    if path[0] != start:
        return []

    return path


# --------------------------------------
# Crop Maze Region
# --------------------------------------
def crop_to_maze(binary):
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]
    return cropped, x, y


# --------------------------------------
# Streamlit App
# --------------------------------------
st.title("PADMAVYUHAM _ A Maze Solver")

uploaded = st.file_uploader("Upload Maze Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Binary conversion
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    maze = cv2.bitwise_not(binary)

    # Crop maze only
    cropped_maze, offset_x, offset_y = crop_to_maze(maze)

    # Block outer border
    cropped_maze[0, :] = 0
    cropped_maze[-1, :] = 0
    cropped_maze[:, 0] = 0
    cropped_maze[:, -1] = 0

    # Click Start
    st.subheader("Click Start Point")
    start_click = streamlit_image_coordinates(image, key="start")

    if start_click:
        sx, sy = start_click["x"], start_click["y"]
        start = (sy - offset_y, sx - offset_x)
        st.success(f"Start Selected: {start}")

        # Click End
        st.subheader("Click End Point")
        end_click = streamlit_image_coordinates(image, key="end")

        if end_click:
            ex, ey = end_click["x"], end_click["y"]
            end = (ey - offset_y, ex - offset_x)
            st.success(f"End Selected: {end}")

            # Solve Exact Center Path
            parent = centered_path(cropped_maze, start, end)
            path = reconstruct(parent, start, end)

            if len(path) == 0:
                st.error("No valid path found")
            else:
                solved = img_np.copy()

                # Draw perfect centered path
                for (r, c) in path:
                    rr = r + offset_y
                    cc = c + offset_x
                    cv2.circle(solved, (cc, rr), 1, (255, 0, 0), -1)

                # Resize output 1.8×
                scale = 1.8
                new_w = int(solved.shape[1] * scale)
                new_h = int(solved.shape[0] * scale)

                solved_big = cv2.resize(solved, (new_w, new_h))

                #st.subheader("Final Perfect Center Path")
                st.image(solved_big)