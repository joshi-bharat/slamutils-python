gyro_noise = 0.00016017
accl_noise = 0.00071376
gyro_random_walk = 0.00000165
acc_random_walk = 0.00002874

multiplier = 5.0
gyro_noise *= multiplier
accl_noise *= multiplier
acc_random_walk *= multiplier
gyro_random_walk *= multiplier

# print(f"IMU.NoiseGyro: {gyro_noise:.8f}")
# print(f"IMU.NoiseAcc: {accl_noise:.8f}")
# print(f"IMU.GyroWalk: {gyro_random_walk:.8f}")
# print(f"IMU.AccWalk: {acc_random_walk:.8f}")

print(
    f"gyroscope_noise_density: {gyro_noise:.8f}"
)  # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
print(
    f"gyroscope_random_walk:  {gyro_random_walk:.8f}"
)  # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
print(
    f"accelerometer_noise_density: {accl_noise:.8f}"
)  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
print(f"accelerometer_random_walk: {acc_random_walk:.8f}")
