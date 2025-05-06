import sys

def clean_file(input_file, output_file):
    # Leggi il contenuto del file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Pulisci le righe:
    # 1. Rimuovi spazi e tabulazioni all'inizio solo se la riga è composta esclusivamente da questi.
    # 2. Rimuovi spazi e tabulazioni alla fine di tutte le righe.
    cleaned_lines = []
    for line in lines:
        if line.strip() == '':
            cleaned_lines.append('')
        else:
            cleaned_lines.append(line.rstrip())

    # Rimuovi più ritorni a capo consecutivi lasciandone solo uno
    final_lines = []
    previous_line_empty = False

    for line in cleaned_lines:
        if line == '':
            if not previous_line_empty:
                final_lines.append(line)
            previous_line_empty = True
        else:
            final_lines.append(line)
            previous_line_empty = False

    # Scrivi il contenuto pulito in un nuovo file (o sovrascrivi quello esistente)
    with open(output_file, 'w') as file:
        for line in final_lines:
            file.write(line + '\n')

# Verifica se sono stati passati i parametri di input e output file
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo: python script.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        clean_file(input_file, output_file)