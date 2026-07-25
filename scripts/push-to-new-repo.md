# Push this scaffold to your new GitHub/GitLab repo

From **this folder** (`nerva-train/`), after you create an empty remote:

```bash
cd nerva-train
git init
git add .
git commit -m "Initial nerva-train product stack extract."
git branch -M main
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

Or copy the whole directory into a clone of the empty repo and commit there.

Do **not** copy the parent monorepo’s `tools/erg_*`, TagWorld, or probe suite into this tree.
