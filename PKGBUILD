# Maintainer: hugo.dn.ferreira@gmail.com
pkgname=p2ascii
pkgver=1.0.0
pkgrel=1
pkgdesc="A simple tool to convert pixel art images to ASCII using OpenCV"
arch=('any')
url="https://github.com/Hugana/p2ascii"
license=('MIT')
depends=('python' 'opencv')
source=("$pkgname-$pkgver.tar.gz::$url/archive/refs/tags/v$pkgver.tar.gz")
sha256sums=('SKIP')

package() {
  cd "$srcdir/$pkgname-$pkgver"

  # Create destination directory
  install -d "$pkgdir/usr/share/$pkgname"

  # Copy script and Images/ directory to /usr/share/p2ascii
  cp -r p2ascii.py Images "$pkgdir/usr/share/$pkgname/"

  # Symlink to make the script callable as 'p2ascii'
  install -d "$pkgdir/usr/bin"
  ln -s "/usr/share/$pkgname/p2ascii.py" "$pkgdir/usr/bin/p2ascii"
}
