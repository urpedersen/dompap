def progress_bar(current_step, total_num_steps, this_stride, finalize=False):
    """Print progress bar to console. """
    from datetime import datetime
    print('\b' * 7, end='')  # Delete previous percent value
    if finalize:
        print('  done')
        return
    if current_step % (this_stride * 50) == 0:
        if current_step > 0:
            print(f'|\n{datetime.now()} {current_step:12,} ', end='')
        else:
            print(f'{datetime.now()} {current_step:12,} ', end='')
    if current_step % (this_stride * 10) == 0:
        print('|', end='')
    elif current_step % this_stride == 0:
        if current_step % (this_stride * 5) == 0:
            print(':', end='')
        else:
            print('.', end='')
    percent_done = current_step / total_num_steps * 100
    print(f' {percent_done:5.1f}%', end='')
