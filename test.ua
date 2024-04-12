# +++ for compatiblity with interpreter
dynamic ← ()
fnsig ← (◌◌)
attrib ← (◌)
use ← (°□□)
# ---

fn = (each(+1 un / *))
fn2 = fn

DoShit ← dynamic(+1) # (this has no reason to be dynamic)
DoShitWrapper ← fnsig["int"]"arr[int]" dynamic(
  DoShit
)

# this is now automatically converted to a template
Inc ← (+1)
IncB ← attrib["inline"](+1)

A ← fnsig["int"]"int"(Inc)
B ← fnsig["int"]"int"(IncB)
C ← fnsig["float"]"float"(Inc)
D ← fnsig["float"]"float"(IncB)
E ← (Inc)
F ← fnsig["int"]"int"(E)
G ← fnsig["dyn"]"int"(use)
H ← fnsig["int"]"int"(G)

# The generated binary should behave somewhat like this:
#
# IntArray DoShitWrapper(int a) {
#   auto interpr = new Interpreter(DoShit__assembly); // pass uasm to interpreter
#   interpr.push(new Dynamic(a));
#   interpr.run();
#   return interpr.pop().as(IntArray); // as throws a runtime exception if unsuccesfull
# }
# int Inc__int(int a) { return a + 1; }
# int Inc__float(double a) { return a + 1; }
# int A(int a) { return Inc__int(a); }
# int B(int a) { return a + 1; }
# float C(float a) { return Inc__float(a); }
# float D(float a) { return a + 1; }
# int E__int(int a) { return Inc__int(a); }
# int F(int a) { return E__int(a); }
# Dynamic G(int a) { return new Dynamic(a); }
# int H(int a) { return G(a).as(IntArray); } // as throws a runtime exception if unsuccesfull

