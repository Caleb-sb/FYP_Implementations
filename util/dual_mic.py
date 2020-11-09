def align(self):
    mic1_arr = self.to_process[0][50000:]
    mic2_arr = self.to_process[1][50000:]

    # print(mic2_arr.size)
    assert (mic1_arr.size == mic2_arr.size and mic1_arr.size > 50400)
    min = 65535
    best_shift = -4000
    slice = mic1_arr[4000:50000-4000]
    for shift in range(-4000, 4000):
        check = mean_squared_error(slice, mic2_arr[4000+shift:50000-4000+shift])
        if (check < min):
            min=check
            best_shift = shift
    print("Shift:", best_shift)
    self.delay = best_shift
