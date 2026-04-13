"""Snake-spill med Pillow-rendering for Gradio."""

import random
from PIL import Image, ImageDraw, ImageFont


class SnakeGame:
    """Snake-spillogikk med relativ styring (left/right) og bilderendering.

    Slangen styres med 'left' og 'right' relativt til fartsretningen.
    'left' svinger alltid mot slangens venstre side, 'right' mot høyre.
    """

    DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    TURN = {
        ("up", "left"): "left",
        ("up", "right"): "right",
        ("down", "left"): "right",
        ("down", "right"): "left",
        ("left", "left"): "down",
        ("left", "right"): "up",
        ("right", "left"): "up",
        ("right", "right"): "down",
    }

    def __init__(self, grid_size: int = 20, cell_size: int = 30) -> None:
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] = (0, 0)
        self.direction: str = "right"
        self.score: int = 0
        self.game_over: bool = False
        self.reset()

    def reset(self) -> None:
        """Nullstill spillet: slange i midten, tilfeldig mat, score = 0."""
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction = "right"
        self.score = 0
        self.game_over = False
        self._spawn_food()

    def _spawn_food(self) -> None:
        """Plasser mat på en tilfeldig rute som ikke er okkupert av slangen."""
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        self.food = random.choice(free) if free else (0, 0)

    def turn(self, relative_direction: str) -> None:
        """Sving slangen relativt til fartsretningen ('left' eller 'right')."""
        if relative_direction in ("left", "right") and not self.game_over:
            self.direction = self.TURN[(self.direction, relative_direction)]

    def step(self) -> dict:
        """Flytt slangen ett steg fremover i nåværende retning."""
        if self.game_over:
            return {"score": self.score, "game_over": True, "direction": self.direction}

        dx, dy = self.DELTAS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.game_over = True
            return {"score": self.score, "game_over": True, "direction": self.direction}

        if new_head in self.snake[:-1]:
            self.game_over = True
            return {"score": self.score, "game_over": True, "direction": self.direction}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()

        return {"score": self.score, "game_over": False, "direction": self.direction}

    def render(self) -> Image.Image:
        """Tegn spillbrettet som et PIL-bilde."""
        header_h = 40
        board_px = self.grid_size * self.cell_size
        img = Image.new("RGB", (board_px, board_px + header_h), "#1a1a2e")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default()

        if self.game_over:
            header_text = f"GAME OVER  |  Score: {self.score}"
        else:
            header_text = f"Score: {self.score}  |  Dir: {self.direction}"
        draw.text((10, 8), header_text, fill="#ffffff", font=font)

        grid_color = "#2a2a4a"
        for i in range(self.grid_size + 1):
            px = i * self.cell_size
            draw.line([(px, header_h), (px, header_h + board_px)], fill=grid_color)
            draw.line([(0, header_h + px), (board_px, header_h + px)], fill=grid_color)

        fx, fy = self.food
        self._draw_cell(draw, fx, fy, header_h, "#ff6b6b")

        for i, (sx, sy) in enumerate(self.snake):
            color = "#00d4aa" if i == 0 else "#00b894"
            self._draw_cell(draw, sx, sy, header_h, color)

        if self.game_over:
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 140))
            img = img.convert("RGBA")
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)
            try:
                big_font = ImageFont.truetype("arial.ttf", 48)
            except OSError:
                big_font = ImageFont.load_default()
            text = "GAME OVER"
            bbox = draw.textbbox((0, 0), text, font=big_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (board_px - tw) // 2
            y = header_h + (board_px - th) // 2
            draw.text((x, y), text, fill="#ff6b6b", font=big_font)
            img = img.convert("RGB")

        return img

    def _draw_cell(self, draw: ImageDraw.Draw, x: int, y: int, header_h: int, color: str) -> None:
        """Tegn én rute med litt padding."""
        pad = 2
        x0 = x * self.cell_size + pad
        y0 = header_h + y * self.cell_size + pad
        x1 = (x + 1) * self.cell_size - pad
        y1 = header_h + (y + 1) * self.cell_size - pad
        draw.rounded_rectangle([x0, y0, x1, y1], radius=4, fill=color)


if __name__ == "__main__":
    game = SnakeGame()
    print(f"Slange: {game.snake}")
    print(f"Mat: {game.food}")
    print(f"Retning: {game.direction}")

    result = game.step()
    print(f"\nEtter step(): {result}")
    print(f"Slange: {game.snake}")

    game.turn("left")
    print(f"\nEtter turn('left'): retning = {game.direction}")
    result = game.step()
    print(f"Etter step(): {result}")
    print(f"Slange: {game.snake}")

    game.turn("right")
    print(f"\nEtter turn('right'): retning = {game.direction}")
    result = game.step()
    print(f"Etter step(): {result}")
    print(f"Slange: {game.snake}")
