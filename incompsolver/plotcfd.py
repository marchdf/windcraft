import numpy as np
import matplotlib.pyplot as plt


def plotflowfields(
    velx,
    vely,
    pres,
    div,
    xmin,
    xmax,
    ymin,
    ymax,
    ncellx,
    ncelly,
    ng,
    lscale,
    vscale,
    pscale,
):
    xmn = xmin * lscale
    xmx = xmax * lscale

    ymn = ymin * lscale
    ymx = ymax * lscale

    dx = (xmx - xmn) / ncellx
    dy = (ymx - ymn) / ncelly

    x = np.arange(xmn + 0.5 * dx, xmx, dx)
    y = np.arange(ymn + 0.5 * dy, ymx, dy)

    Nx = len(x)
    Ny = len(y)

    X, Y = np.meshgrid(x, y)

    velx_cc = np.zeros((ncelly, ncellx))
    vely_cc = np.zeros((ncelly, ncellx))
    pres_cc = np.zeros((ncelly, ncellx))
    div_cc = np.zeros((ncelly, ncellx))

    for j in range(ncelly):
        for i in range(ncellx):
            velx_cc[j][i] = 0.5 * (velx[j + ng][i + ng] + velx[j + ng][i + ng + 1])
            vely_cc[j][i] = 0.5 * (vely[j + ng][i + ng] + vely[j + ng + 1][i + ng])
            pres_cc[j][i] = pres[j + ng][i + ng]
            div_cc[j][i] = div[j + ng][i + ng]

    plt.figure(1)
    plt.clf()
    CFplot2 = plt.contourf(X, Y, vely_cc[:][:] * vscale, 30)
    cbar2 = plt.colorbar(CFplot2)
    plt.title("y velocity")

    plt.figure(2)
    plt.clf()
    CFplot3 = plt.contourf(X, Y, pres_cc[:][:] * pscale, 30)
    plt.quiver(X, Y, velx_cc[:][:] * vscale, vely_cc[:][:] * vscale)
    cbar3 = plt.colorbar(CFplot3)
    plt.title("pressure")

    plt.figure(3)
    plt.clf()
    CFplot1 = plt.contourf(X, Y, velx_cc[:][:] * vscale, 30)
    cbar1 = plt.colorbar(CFplot1)
    plt.title("x velocity")

    plt.figure(4)
    plt.clf()
    CFplot2 = plt.contourf(X, Y, div_cc[:][:] * vscale, 30)
    cbar2 = plt.colorbar(CFplot2)
    plt.title("div u")
    plt.show()
