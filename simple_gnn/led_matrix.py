#!/usr/bin/env python3
import colorsys
import random
import signal
import sys
import time
import matplotlib.cm as cm

import numpy as np
from rpi_ws281x import Color, PixelStrip

# LED strip configuration:
LED_HEIGHT = 4        # Number of LED pixel rows.
LED_WIDTH = 8         # Number of LED pixel columns.
LED_COUNT = LED_HEIGHT * LED_WIDTH
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53


class LEDMatrix(object):
    def __init__(self):
        self.config = {
            'count': LED_COUNT,
            'width': LED_WIDTH,
            'height': LED_HEIGHT,
        }

        # Create NeoPixel object with appropriate configuration.
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        # Intialize the library (must be called once before other functions).
        self.strip.begin()

        signal.signal(signal.SIGINT, self.exit)

        self.init_animation()
        print("LED strip is ready to receive commands.")

    def init_animation(self):
        for color in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0, 0, 0)]:
            for i in range(self.strip.numPixels()):
                self.strip.setPixelColor(i, Color(*color))
                self.strip.show()
                time.sleep(0.01)

    def set_all(self, color, color_space='rgb'):
        if color_space == 'hsv':
            color = self.hsv2rgb(*color)
        pixels = [color] * self.config['count']
        self._set_pixels(pixels)

    def set_square(self, color, size, top_left, color_space='rgb'):
        if color_space == 'hsv':
            color = self.hsv2rgb(*color)

        if not (top_left[0] >= 0 and top_left[1] >= 0
                and top_left[0] + size <= self.config['height']
                and top_left[1] + size <= self.config['width']):
            print("WARNING: Can't set the desired square.")

        pixels = [0] * self.config['count']
        for i in range(self.config['height']):
            for j in range(self.config['width']):
                if (i >= top_left[0] and i < top_left[0] + size
                        and j >= top_left[1] and j < top_left[1] + size):
                    pixels[i * self.config['width'] + j] = color
                else:
                    pixels[i * self.config['width'] + j] = (0, 0, 0)
        self._set_pixels(pixels)

    def set_individual(self, array, color_space='rgb'):
        if len(array) == self.config['height']:
            pixels = [val for subarray in array for val in subarray]
        else:
            pixels = array

        if color_space == 'hsv':
            pixels = [self.hsv2rgb(*val) for val in pixels]
        self._set_pixels(pixels)

    def set_rainbow(self, brightness, start=0):
        pixels = [self.hsv2rgb(h, 255, brightness)
                  for h in np.roll(np.linspace(0, 255, self.config['count']), start)]
        self._set_pixels(pixels)

    def set_random(self, brightness):
        pixels = [self.hsv2rgb(random.random() * 255, 25, brightness) for i in range(self.config['count'])]
        self._set_pixels(pixels)

    def add_line(self, pixels, index, color, color_space='rgb'):
        if index < 0 or index >= self.config['width']:
            print("ERROR: Index out of range.")
            return

        if color_space == 'hsv':
            color = self.hsv2rgb(*color)

        for i in range(self.config['height']):
            pixels[i * self.config['width'] + index] = color

        return pixels

    def set_line(self, index, color, color_space='rgb'):
        pixels = [(0, 0, 0)] * self.config['count']
        pixels = self.add_line(pixels, index, color, color_space)

        self._set_pixels(pixels)

    def set_percentage(self, percent, resolution, base_color, mode='l2r', color_space='rgb'):
        """
        Set the percentage of the LED strip with a specific color.

        :param percent: Percentage to set (0 to 1).
        :param resolution: Total number of steps. It is divided by the number of bars,
                           and intermediate values are displayed as changes in saturation.
        :param base_color: Base color to use for the percentage.
        :param mode: Mode of the percentage ('l2r' for left to right, 'center' for centered).
        :param color_space: Color space of the base color ('rgb' or 'hsv').
        """
        if color_space == 'rgb':
            base_color = self.rgb2hsv(*base_color)

        if mode == 'center':
            resolution = int(resolution / 2)

        levels = resolution // self.config['width']

        # I think the minus 1 here and plus 1 below is prioritize higher ranges of discrete values over lower ones.
        # This way, when the percent is close to 1, it will be more likely to light up the last LED. This in turn causes
        # that for percent 0, the first bar is dimly lit. The decision was to show even 0 percent as color to signal
        # that LEDs are working.
        if mode == 'l2r':
            max_val = levels * self.config['width'] - 1
        elif mode == 'center':
            max_val = levels * (self.config['width'] // 2) - 1

        pixels = [(0, 0, 0)] * self.config['count']

        x = self.interp(percent, 0, 1, 0, max_val)

        for i in range(x // levels + 1):
            if i < x // levels:
                color = base_color
            else:
                step = (x % levels + 1) / levels
                color = (base_color[0], base_color[1], int(round(base_color[2] * step)))

            if mode == 'l2r':
                pixels = self.add_line(pixels, i, color, color_space='hsv')
            elif mode == 'center':
                pixels = self.add_line(pixels, i + self.config['width'] // 2, color, color_space='hsv')
                pixels = self.add_line(pixels, -i + self.config['width'] // 2 - 1, color, color_space='hsv')

        self._set_pixels(pixels)

    def off(self):
        """Turn off all LEDs."""
        pixels = [(0, 0, 0)] * self.config['count']
        self._set_pixels(pixels)

    def _set_pixels(self, pixels):
        for i, color in enumerate(pixels):
            self.strip.setPixelColor(i, Color(*color))
        self.strip.show()

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h / 255, s / 255, v / 255))

    @staticmethod
    def rgb2hsv(r, g, b):
        return tuple(int(round(i * 255)) for i in colorsys.rgb_to_hsv(r / 255, g / 255, b / 255))

    @staticmethod
    def interp(x, in_min, in_max, out_min, out_max):
        x = max(min(x, in_max), in_min)
        return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    @staticmethod
    def colormap_to_rgb(value: float, cmap_name: str = "cool"):
        cmap = cm.get_cmap(cmap_name)
        r, g, b, _ = cmap(value)
        return (int(r * 255), int(g * 255), int(b * 255))

    def exit(self, signum=None, frame=None):
        print("Turning off LEDs and exiting.")
        for i in range(LED_COUNT):
            self.strip.setPixelColor(i, Color(0, 0, 0))
        self.strip.show()
        sys.exit(0)


if __name__ == '__main__':
    led = LEDMatrix()

    print(f"RGB")
    led.set_all((255, 0, 0))  # Set all LEDs to red
    time.sleep(1)
    led.set_all((0, 255, 0))  # Set all LEDs to green
    time.sleep(1)
    led.set_all((0, 0, 255))  # Set all LEDs to blue
    time.sleep(1)
    led.set_all((0, 0, 0))  # Turn off all LEDs
    time.sleep(2)

    print(f"Squares")
    for i in range(LED_WIDTH - 1):
        led.set_square((255, 255, 0), 2, (0, i))  # Set a yellow square
        time.sleep(0.2)
    for i in range(LED_WIDTH - 2, -1, -1):
        led.set_square((0, 255, 255), 2, (2, i))  # Set a cyan square
        time.sleep(0.2)
    time.sleep(2)

    print("Rainbow")
    for i in range(255):
        led.set_rainbow(i)
        time.sleep(0.01)
    time.sleep(2)

    print("Random")
    led.set_random(100)  # Set random colors with brightness 100
    time.sleep(2)

    print("Percentage")
    led.off()
    for i in range(100):
        led.set_percentage(i / 100, 100, (255, 0, 255), 'l2r')
        time.sleep(0.1)
    time.sleep(2)

    print("Done")
    led.exit()