import { useSearchParams } from "react-router-dom";
import { Boxes, Database, LogOut, RefreshCcw, Upload, Users } from "lucide-react";
import { useAuth } from "../../auth";
import { PageHead, Segmented } from "../../components/ui";
import UploadPanel from "./UploadPanel";
import RecordsPanel from "./RecordsPanel";
import RetrainPanel from "./RetrainPanel";
import ModelsPanel from "./ModelsPanel";
import UsersPanel from "./UsersPanel";

type Tab = "upload" | "records" | "retrain" | "models" | "users";

export default function Admin() {
  const { signOut, user } = useAuth();
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as Tab) || "upload";
  const setTab = (t: Tab) => setParams({ tab: t }, { replace: true });

  return (
    <div className="container">
      <PageHead
        eyebrow="Admin console"
        title="Data, retraining & models"
        actions={<button className="btn sm" onClick={signOut}><LogOut size={15} /> Sign out</button>}
      >
        {user ? <>Signed in as <strong>{user.username}</strong>. </> : null}Add labelled images to the training database, retrain the
        active model with them, manage which model version is served, and manage user accounts.
      </PageHead>
      <div style={{ marginBottom: 18, overflowX: "auto" }}>
        <Segmented label="Admin section" value={tab} onChange={setTab} options={[
          { value: "upload", label: <><Upload size={15} /> Upload</> },
          { value: "records", label: <><Database size={15} /> Database</> },
          { value: "retrain", label: <><RefreshCcw size={15} /> Retraining</> },
          { value: "models", label: <><Boxes size={15} /> Models</> },
          { value: "users", label: <><Users size={15} /> Users</> },
        ]} />
      </div>
      {tab === "upload" && <UploadPanel onDone={() => setTab("records")} />}
      {tab === "records" && <RecordsPanel />}
      {tab === "retrain" && <RetrainPanel />}
      {tab === "models" && <ModelsPanel />}
      {tab === "users" && <UsersPanel />}
    </div>
  );
}
