from main import main
import os

if __name__ == '__main__':
    write_head = False
    with open("evaluation_log.csv", "w") as f:
        for root, dirs, files in os.walk("automata"):
            for file in sorted(files):
                eval_log = main(os.path.join(root, file), need_plot=False)
                key_line = "name"
                line = file.split('.json')[0]
                eval_log.pop('name')
                for key, val in eval_log.items():
                    if key == "time":
                        for name, time in val:
                            key_line += ("," + name + "_time")
                            line += ("," + str(time))
                        continue
                    key_line += ("," + key)
                    line += ("," + str(val))
                if not write_head:
                    f.write(key_line + "\n")
                    write_head = True
                f.write(line + "\n")
        f.close()
