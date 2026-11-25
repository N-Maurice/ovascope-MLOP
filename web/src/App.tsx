import { lazy, Suspense, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Skeleton } from "./components/ui";
import { useAuth } from "./auth";
import Home from "./pages/Home";

const Classify = lazy(() => import("./pages/Classify"));
const Classes = lazy(() => import("./pages/Classes"));
const Dataset = lazy(() => import("./pages/Dataset"));
const Performance = lazy(() => import("./pages/Performance"));
const Status = lazy(() => import("./pages/Status"));
const About = lazy(() => import("./pages/About"));
const SignIn = lazy(() => import("./pages/SignIn"));
const Admin = lazy(() => import("./pages/admin/Admin"));
const Account = lazy(() => import("./pages/Account"));
const NoAdminAccess = lazy(() => import("./pages/Account").then((m) => ({ default: m.NoAdminAccess })));

function Loading() {
  return (
    <div className="container" style={{ paddingTop: 32 }}>
      <Skeleton height={40} style={{ width: 280, marginBottom: 16 }} />
      <Skeleton height={260} />
    </div>
  );
}

function RequireAdmin({ children }: { children: ReactNode }) {
  const { user, viaKey, isAdmin, checking } = useAuth();
  const location = useLocation();
  if (checking) return <Loading />;
  if (!user && !viaKey) return <Navigate to="/sign-in" replace state={{ from: location.pathname + location.search }} />;
  return isAdmin ? children : <NoAdminAccess />;
}

function NotFound() {
  return (
    <div className="container empty" style={{ paddingTop: 80 }}>
      <h2>Page not found</h2>
      <p>This page does not exist.</p>
      <Link className="btn" to="/">Go home</Link>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/classify" element={<Classify />} />
          <Route path="/classes" element={<Classes />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/status" element={<Status />} />
          <Route path="/about" element={<About />} />
          <Route path="/sign-in" element={<SignIn />} />
          <Route path="/account" element={<Account />} />
          <Route path="/admin" element={<RequireAdmin><Admin /></RequireAdmin>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
