#
# Copyright (C) 2018 - Massachusetts Institute of Technology (MIT)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Quaternion transformations"""

import math
from tsig.util.pointing import to_quaternion
from tsig.util.rdr import normalize_ra, normalize_dec, normalize_roll

DEG_TO_RAD = math.pi / 180.0


def quat_mult(quata, quatb):
    w1, x1, y1, z1 = quata
    w2, x2, y2, z2 = quatb
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def perturb_angles(ra, dec, roll, quat, xform=None):
    """
    Apply perturbation to specified angles using the quaternion.
    Angles ra, dec, and roll are in degrees.
    Quaternion is of the form w, x, y, z in radians.
    """
    if xform is None:
        xform = QuaternionTransform
    q_old = xform.rdr_to_quat((ra, dec, roll))
    q_new = quat_mult(q_old, quat)
    ra, dec, roll = xform.quat_to_rdr(q_new)
    return ra, dec, roll


class LevineTransform(object):
    """
    Based on the C implementation by Al Levine, September 2017.  The names
    here basically follow those in the C code.
    """

    @staticmethod
    def rdr_to_quat(rdr, coordsys='EQU'):
        """Convert ra, dec, roll (in degrees) to quaternion (w,x,y,z)"""
        rmat1 = LevineTransform.sky_to_sc_mat(rdr)
        if coordsys == 'EQU':
            rmat3 = LevineTransform.create_matrix()
            for i in range(3):
                for j in range(3):
                    rmat3[i][j] = rmat1[i][j]
        else:
            eul = [0.0, -23.439 * DEG_TO_RAD, 0.0]
            rmat2 = LevineTransform.eulerm313(eul)
            rmat5 = LevineTransform.trans(rmat2)
            rmat3 = LevineTransform.matmat(rmat1, rmat5)
        qa = LevineTransform.matrix_to_quat(rmat3)
        q = LevineTransform.quat_sign(qa)
        return q[3], q[0], q[1], q[2]

    @staticmethod
    def quat_to_rdr(q):
        """Convert quaternion (w,x,y,z) to ra, dec, roll (in degrees)"""
        quat = (q[1], q[2], q[3], q[0])
        rmat1 = LevineTransform.quatmat(quat)
        ea = LevineTransform.mateuler323(rmat1)
        ra = normalize_ra(ea[0])
        dec = normalize_dec(math.pi / 2 - ea[1])
        roll = normalize_roll(ea[2] - math.pi)
        return map(math.degrees, (ra, dec, roll))

    @staticmethod
    def create_matrix(ncol=3, nrow=3):
        mat = []
        for i in range(nrow):
            mat.append([0] * ncol)
        return mat

    @staticmethod
    def slewang(mata, matb):
        """
        For two specified rigid body orientations, compute the rotation axis
        and rotation angle to transform from the first to the second
        orientation.

        The two orientations are specified by rotation matrices.

        Each matrix is defined so that the matrix multiplies a vector with
        celestial components to produce a vector with body components.
        Therefore, the first row of each matrix represents the celestial
        components of the body x-axis, and so forth.
        """
        eps = 1.0e-8
        angle = 0
        eigvec = [0] * 3
        dfsqlen = [0] * 3
        diff = LevineTransform.create_matrix()
        for i in range(3):
            for j in range(3):
                diff[i][j] = mata[i][j] - matb[i][j]
            dfsqlen[i] = LevineTransform.dot(diff[i], diff[i])
        xmax = 0.0
        xmin = 5.0
        imax = -1
        imin = -1
        for i in range(3):
            if dfsqlen[i] > xmax:
                xmax = dfsqlen[i]
                imax = i
            if dfsqlen[i] < xmin:
                xmin = dfsqlen[i]
                imin = i
        if imax == -1 or xmax < eps * eps:
            angle = 0.0
        else:
            imid = 3 - (imax + imin)
            eigvec = LevineTransform.dcross(diff[imax], diff[imid])
            eigvec = LevineTransform.dnorm(eigvec)
            yp = LevineTransform.dcross(eigvec, mata[imax])
            yp = LevineTransform.dnorm(yp)
            xp = LevineTransform.dcross(yp, eigvec)
            xp = LevineTransform.dnorm(xp)
            bx = LevineTransform.dot(matb[imax], xp)
            by = LevineTransform.dot(matb[imax], yp)
            angle = math.atan2(by, bx)
        return angle, eigvec

    @staticmethod
    def matrix_to_quat(mat):
        """Construct a quaternion from a rotation matrix"""
        matid = LevineTransform.create_matrix()
        for i in range(3):
            for j in range(3):
                matid[i][j] = 0.0
                if i == j:
                    matid[i][j] = 1.0
        angle, eigvec = LevineTransform.slewang(matid, mat)
        s = math.sin(angle / 2.0)
        q = [0] * 4
        for i in range(3):
            q[i] = eigvec[i] * s
        q[3] = math.cos(angle / 2.0)
        return q

    @staticmethod
    def sky_to_sc_mat(rdr):
        """
        Multiply a 3-vector in celestial coordinates to get a 3-vector in
        spacecraft coordinates.
        """
        xeul = [0] * 3
        xeul[0] = DEG_TO_RAD * rdr[0]
        xeul[1] = math.pi / 2.0 - DEG_TO_RAD * rdr[1]
        xeul[2] = DEG_TO_RAD * rdr[2]
        xeul[2] += math.pi
        rmat1 = LevineTransform.eulerm323(xeul)
        return rmat1

    @staticmethod
    def quat_sign(qin):
        """
        Make the fourth component non-negative by multiplying all components
        by -1 if the component is negative.
        """
        qout = [0] * 4
        for i in range(4):
            if qin[3] < 0.0:
                qout[i] = -qin[i]
            else:
                qout[i] = qin[i]
        return qout

    @staticmethod
    def quatnorm(q):
        """Normalize a quaternion"""
        qsq = 0.0
        for i in range(4):
            qsq += q[i] * q[i]
        qsq = math.sqrt(qsq)
        qout = [0] * 4
        for i in range(4):
            qout[i] = q[i] / qsq
        return qout

    @staticmethod
    def quatmult(q, qp):
        """
        Quaternion multiplication representing composition of two rotations
        q  - first rotation
        qp - second rotation
        A(q'') = A(q') x A(q)
        """
        qpp = [0] * 4
        qpp[0] = q[0]*qp[3]+q[1]*qp[2]-q[2]*qp[1]+q[3]*qp[0];
        qpp[1] = -q[0]*qp[2]+q[1]*qp[3]+q[2]*qp[0]+q[3]*qp[1];
        qpp[2] = q[0]*qp[1]-q[1]*qp[0]+q[2]*qp[3]+q[3]*qp[2];
        qpp[3] = -q[0]*qp[0]-q[1]*qp[1]-q[2]*qp[2]+q[3]*qp[3];
        return qpp

    @staticmethod
    def quatmat(q):
        """
        Construct a rotation matrix from a quaternion.

        Rotation matrix multiplies vectors with celestial components to get
        spacecraft components.
        """
        mat = LevineTransform.create_matrix()
        mat[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]
        mat[0][1] = 2.0*(q[0]*q[1]+q[2]*q[3])
        mat[0][2] = 2.0*(q[0]*q[2]-q[1]*q[3])
        mat[1][0] = 2.0*(q[0]*q[1]-q[2]*q[3])
        mat[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3]
        mat[1][2] = 2.0*(q[1]*q[2]+q[0]*q[3])
        mat[2][0] = 2.0*(q[0]*q[2]+q[1]*q[3])
        mat[2][1] = 2.0*(q[1]*q[2]-q[0]*q[3])
        mat[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3]
        return mat

    @staticmethod
    def mateuler323(mat):
        """
        Construct 3-2-3 Euler angles from a rotation matrix.

        The Euler angles specify the orientation of a new coordinate system
        relative to an old coordinate system.

        The rotation matrix multiplies vectors expressed in the original
        coordinate system to give components in the new coordinate system.

        Matrix form is from "Spacecraft Attitude Determination and Contrl",
        ed J. Wertz

        NB 03sep2017: This needs a fix for cases where the new z axis is
        parallel or antiparallel to the old z axis.  The precision of how
        accurately this is known may still be an issue.
        """
        zp = [0] * 3
        zi = [0] * 3
        for i in range(3):
            zp[i] = mat[2][i]
            zi[i] = mat[i][2]
        xlng, xlat = LevineTransform.dcrsph(zp)
        euler = [0] * 3
        euler[0] = xlng
        if euler[0] > math.pi:
            euler[0] -= 2.0 * math.pi
        if euler[0] < -math.pi:
            euler[0] += 2.0 * math.pi
        euler[1] = math.pi / 2 - xlat
        if xlat == math.pi / 2:
            for i in range(3):
                zi[i] = mat[0][i]
            xlng, xlat = LevineTransform.dcrsph(zi)
            euler[2] = xlng
        elif xlat == -math.pi / 2:
            for i in range(3):
                zi[i] = mat[0][i]
            xlng, xlat = LevineTransform.dcrsph(zi)
            euler[2] = -xlng
        else:
            xlng, xlat = LevineTransform.dcrsph(zi)
            euler[2] = -xlng + math.pi
        if euler[2] > math.pi:
            euler[2] -= 2.0 * math.pi
        if euler[2] < -math.pi:
            euler[2] += 2.0 * math.pi
        return euler

    @staticmethod
    def eulerm313(euler):
        """
        Construct rotation matrix from 3-1-3 Euler angles.

        The Euler angles specify the orientation of a new coordinate system
        relative to an old coordinate system.

        The rotation matrix multiplies vectors expressed in the original
        coordinate system to give components in the new coordinate system.
        """
        mat1 = LevineTransform.rotm1(2, euler[0])
        mat2 = LevineTransform.rotm1(0, euler[1])
        mata = LevineTransform.matmat(mat2, mat1)
        mat1 = LevineTransform.rotm1(2, euler[2])
        mat = LevineTransform.matmat(mat1, mata)
        return mat

    @staticmethod
    def eulerm323(euler):
        """
        Construct rotation matrix from 3-2-3 Euler angles.

        The Euler angles specify the orientation of a new coordinate system
        relative to an old coordinate system.

        The rotation matrix multiplies vectors expressed in the original
        coordinate system to give components in the new coordinate system.
        """
        mat1 = LevineTransform.rotm1(2, euler[0])
        mat2 = LevineTransform.rotm1(1, euler[1])
        mata = LevineTransform.matmat(mat2, mat1)
        mat1 = LevineTransform.rotm1(2, euler[2])
        mat = LevineTransform.matmat(mat1, mata)
        return mat

    @staticmethod
    def euler313_to_quat(ea):
        """
        Construct a quaternion from 3-1-3 Euler angles (phi, theta, psi)
        All angles in radians
        """
        phi = ea[0]
        theta = ea[1]
        psi = ea[2]
        sth = math.sin(theta / 2.0)
        cth = math.cos(theta / 2.0)
        stmps = math.sin((phi - psi) / 2.0)
        ctmps = math.cos((phi - psi) / 2.0)
        stpps = math.sin((phi + psi) / 2.0)
        ctpps = math.cos((phi + psi) / 2.0)
        q = [0] * 4
        q[0] = sth * ctmps
        q[1] = sth * stmps
        q[2] = cth * stpps
        q[3] = cth * ctpps
        return LevineTransform.quatnorm(q)

    @staticmethod
    def euler323_to_quat(ea):
        """
        Construct a quaternion from 3-2-3 Euler angles (phi, theta, psi)
        All angles in radians
        """
        q1 = [0] * 4
        q1[0] = 0.0
        q1[1] = 0.0
        q1[2] = math.sin(ea[0] / 2.0)
        q1[3] = math.cos(ea[0] / 2.0)
        q2 = [0] * 4
        q2[0] = 0.0
        q2[1] = math.sin(ea[1] / 2.0)
        q2[2] = 0.0
        q2[3] = math.cos(ea[1] / 2.0)
        q3 = [0] * 4
        q3[0] = 0.0
        q3[1] = 0.0
        q3[2] = math.sin(ea[2] / 2.0)
        q3[3] = math.cos(ea[2] / 2.0)
        q12 = LevineTransform.quatmult(q1, q2)
        q = LevineTransform.quatmult(q12, q3)
        return q

    @staticmethod
    def rotm1(nax, angle):
        """
        Construct a rotation matrix that will rotate the coordinate system
        about one of the coordinate axes by an angle (in radians).

        nax = 0, 1, or 2
        
        The rotation matrix muliplies vectors expressed in the original
        coordinate system to give components in the new coordinate system.

        Alternatively, the matrix rotates the vector by -angle around the
        given axis.
        """
        mat = LevineTransform.create_matrix()
        n1 = nax
        n2 = (n1 + 1) % 3
        n3 = (n2 + 1) % 3
        s = math.sin(angle)
        mat[n1][n1] = 1.0
        mat[n3][n3] = mat[n2][n2] = math.cos(angle)
        mat[n2][n3] = s
        mat[n3][n2] = -s
        return mat

    @staticmethod
    def matmat(mata, matb):
        """Multiply two matrices"""
        mat = LevineTransform.create_matrix()
        for i in range(3):
            for j in range(3):
                mat[i][j] = 0.0
                for k in range(3):
                    mat[i][j] += mata[i][k] * matb[k][j]
        return mat

    @staticmethod
    def matvec(mat, vec):
        """Multiply a matrix by a vector"""
        v = [0] * 3
        for i in range(3):
            for j in range(3):
                v[i] += mat[i][j] * v[j]
        return v

    @staticmethod
    def trans(mat):
        """Construct the transpose of a matrix"""
        tmat = LevineTransform.create_matrix()
        for i in range(3):
            for j in range(3):
                tmat[i][j] = mat[j][i]
        return tmat

    @staticmethod
    def dcross(a, b):
        """Cross product of 2 3-vectors"""
        c = [0] * 3
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
        return c

    @staticmethod
    def dnorm(vec):
        """Normalize a 3-vector to be of unit length"""
        length = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
        length = math.sqrt(length)
        v = [x for x in vec]
        if length > 0:
            for i in range(3):
                v[i] = vec[i] / length
        return v

    @staticmethod
    def dot(a, b):
        """Dot product of two vectors"""
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def dcrsph(vec):
        """
        Convert cartesian vector to spherical coordinates
        The returned ra,dec are in radians
        """
        ra = 0.0
        dec = 0.0
        length = math.sqrt(LevineTransform.dot(vec, vec))
        if length > 0.0:
            dec = math.asin(vec[2] / length)
            if vec[0] != 0.0 or vec[1] != 0.0:
                ra = math.atan2(vec[1], vec[0])
                if ra < 0.0:
                    ra += 2.0 * math.pi
        return ra, dec


class KephartTransform(object):

    @staticmethod
    def rdr_to_quat(rdr):
        """Convert ra, dec, roll in degrees to a quaternion w,x,y,z"""
        # This was never properly implemented in the POC code, so use Nguyen
#        ra = rdr[0]
#        dec = rdr[1]
#        roll = rdr[2]
#        import spiceypy as spice
#        return spice.m2q(spice.eul2m(ra, dec, roll, 3, 2, 3))
        return NguyenTransform.rdr_to_quat(rdr)

    @staticmethod
    def quat_to_rdr(quat):
        """
        Convert a quaternion to TESS ra, dec, roll angles

        The quaternion is in the form w, x, y, z where x, y, and z are angles
        in radians around the respective x, y, and z axes, and w is the
        quaternion angle, also in radians.

        Returns a tuple of angles ra, dec, roll in degrees.

        The roll is defined as the rotation from the line of RA - this is not
        the same as an intrinsic rotation angle, so it must be shifted by 180
        degrees.  Similarly, the declination must be subtracted from 90 degrees
        in order to obtain the rotation around the Y axis.  The RA is
        normalized to [0, 2*pi].
        """
        # NB: the kephart implementation uses x,y,z,w but this is w,x,y,z

        q = to_quaternion(quat)
        if q is None:
            raise TypeError("Cannot get a quaternion from %s" % quat)

        # get ra/dec/roll from euler angles (ZYZ)
        import spiceypy as spice
        rdr = spice.m2eul(spice.invert(spice.q2m(q)), 3, 2, 3)

        # ensure that the translated values are within proper ranges
        fail = []
        if not (-math.pi <= rdr[0] <= math.pi):
            fail.append("roll: %s" % rdr[0])
        if not (-math.pi <= rdr[2] <= math.pi):
            fail.append("right ascension: %s" % rdr[2])
        if not (0 <= rdr[1] <= math.pi):
            fail.append("declination: %s" % rdr[1])
        if fail:
            raise TypeError("Value out of range: %s" % ';'.join(fail))

        # adjust for TESS-specific usage
        ra = normalize_ra(rdr[2])
        dec = normalize_dec(math.pi / 2.0 - rdr[1])
        roll = normalize_roll(rdr[0] - math.pi)

        return map(math.degrees, (ra, dec, roll))


class NguyenTransform(object):

    @staticmethod
    def rdr_to_mat(rdr):
        ra = rdr[0]
        dec = rdr[1]
        roll = rdr[2]
        import spiceypy as spice
        roll = roll + 180.0
        eul = map(math.radians, (ra, 90 - dec, roll))
        m = spice.eul2m(eul[2], eul[1], eul[0], 3, 2, 3)
        m = spice.invert(m)
        return m

    @staticmethod
    def rdr_to_quat(rdr):
        """Convert ra,dec,roll (in degrees) to quaternion (w,x,y,z)"""
        import spiceypy as spice
        m = NguyenTransform.rdr_to_mat(rdr)
        q = spice.m2q(m)
        return q[0], q[1], q[2], q[3]

    @staticmethod
    def quat_to_rdr(quat):
        """Convert quaternion (w,x,y,z) to ra,dec,roll (in degrees)"""
        import spiceypy as spice
        try:
            quat = [float(x) for x in quat]
        except ValueError:
            raise TypeError
        rdr = spice.m2eul(spice.invert(spice.q2m(quat)), 3, 2, 3)
        ra = normalize_ra(rdr[2])
        dec = normalize_dec(math.pi / 2 - rdr[1])
        roll = normalize_roll(rdr[0] - math.pi)
        return map(math.degrees, (ra, dec, roll))


# define the default transformation method.  use the levine transform so that
# there is no unnecessary spice dependency.
QuaternionTransform = LevineTransform

TRANSFORMS = {
    'levine': LevineTransform,
    'kephart': KephartTransform,
    'nguyen': NguyenTransform,
}
