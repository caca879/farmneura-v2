/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'sans-serif'],
      },
      colors: {
        farmGreen: {
          50: '#e8f5e9',
          100: '#c8e6c9',
          500: '#2e7d32',
          700: '#1b5e20',
          800: '#1b4d3e',
          900: '#0d382c',
        },
        warmOrange: {
          500: '#e65100',
          600: '#d84315',
          700: '#bf360c',
        }
      }
    },
  },
  plugins: [],
}
