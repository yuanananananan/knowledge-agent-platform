// 文件用途：处理数据集上传、解压、安全校验、预览、分页查询和删除。
package businessapi

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/gin-gonic/gin"
)

// listImages 查询当前用户可访问的数据集图片记录。
func (s *Server) listImages(c *gin.Context) {
	rows, err := s.db.QueryContext(c, `SELECT id,dataset_id,name,modality,path,label FROM images WHERE dataset_id=? AND isDelete=0 ORDER BY id`, c.Param("id"))
	writeRows(c, rows, err)
}

// imageFile 校验文件归属后通过 Python 渲染并返回图片内容。
func (s *Server) imageFile(c *gin.Context) {
	var stored string
	if s.db.QueryRowContext(c, `SELECT path FROM images WHERE path=? AND isDelete=0`, c.Query("image_path")).Scan(&stored) != nil {
		c.JSON(404, gin.H{"error": "Image not found"})
		return
	}
	path, err := safePath(s.dataRoot, stored)
	if err != nil {
		c.JSON(403, gin.H{"error": "invalid path"})
		return
	}
	if strings.EqualFold(filepath.Ext(path), ".npy") {
		image, err := s.compute.RenderImage(c.Request.Context(), path)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "image_conversion_failed"})
			return
		}
		c.Header("Content-Disposition", `inline; filename="`+image.Filename+`"`)
		c.Data(http.StatusOK, image.ContentType, image.Content)
		return
	}
	c.File(path)
}

// safeNamePattern 是上传目录名和文件名允许使用的字符白名单。
var safeNamePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,128}$`)

// uploadDataset 保存上传压缩包、安全解压、检查结构并写入数据集及图片记录。
func (s *Server) uploadDataset(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 2<<30)
	modality := strings.ToUpper(strings.TrimSpace(c.PostForm("modality")))
	if modality != "SAR" && modality != "RD" && modality != "MON" {
		c.JSON(400, gin.H{"error": "modality must be SAR, RD or MON"})
		return
	}
	file, header, err := c.Request.FormFile("file")
	if err != nil || !strings.HasSuffix(strings.ToLower(header.Filename), ".zip") {
		c.JSON(400, gin.H{"error": "a ZIP file is required"})
		return
	}
	defer file.Close()
	name := strings.TrimSuffix(filepath.Base(header.Filename), filepath.Ext(header.Filename))
	if supplied := strings.TrimSpace(c.PostForm("name")); supplied != "" {
		name = supplied
	}
	if !safeNamePattern.MatchString(name) {
		c.JSON(400, gin.H{"error": "invalid dataset name"})
		return
	}
	var exists int
	if s.db.QueryRowContext(c, `SELECT 1 FROM datasets WHERE name=? AND isDelete=0`, name).Scan(&exists) == nil {
		c.JSON(409, gin.H{"error": "dataset already exists"})
		return
	}
	if err := os.MkdirAll(s.dataRoot, 0750); err != nil {
		fail(c, err)
		return
	}
	tmp, err := os.CreateTemp("", "knowledge-dataset-*.zip")
	if err != nil {
		fail(c, err)
		return
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = io.Copy(tmp, file); err != nil {
		tmp.Close()
		fail(c, err)
		return
	}
	if err = tmp.Close(); err != nil {
		fail(c, err)
		return
	}
	destination := filepath.Join(s.dataRoot, name)
	if _, err = os.Stat(destination); !os.IsNotExist(err) {
		c.JSON(409, gin.H{"error": "dataset directory already exists"})
		return
	}
	if err = extractZIP(tmpName, destination, name); err != nil {
		_ = os.RemoveAll(destination)
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	keepDatasetFiles := false
	defer func() {
		if !keepDatasetFiles {
			_ = os.RemoveAll(destination)
		}
	}()
	labels, images, err := inspectDataset(destination)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	labelsJSON, _ := json.Marshal(labels)
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(c, `INSERT INTO datasets(name,modality,path,labels,isDelete) VALUES(?,?,?,?,0)`, name, modality, destination, labelsJSON)
	if err != nil {
		fail(c, err)
		return
	}
	datasetID, _ := res.LastInsertId()
	for _, image := range images {
		_, err = tx.ExecContext(c, `INSERT INTO images(dataset_id,name,modality,path,label,split,isDelete) VALUES(?,?,?,?,?,?,0)`, datasetID, name, modality, image.path, image.label, image.split)
		if err != nil {
			fail(c, err)
			return
		}
	}
	if err = tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	keepDatasetFiles = true
	c.JSON(201, gin.H{"message": "Dataset uploaded successfully", "id": datasetID, "name": name, "modality": modality, "path": destination, "labels": labels, "insert_result": gin.H{"inserted": len(images)}})
}

// imageRecord 是扫描数据集目录时使用的内部图片描述。
type imageRecord struct{ path, label, split string }

// extractZIP 在限制路径穿越和根目录的前提下解压数据集。
func extractZIP(archive, destination, expectedRoot string) error {
	reader, err := zip.OpenReader(archive)
	if err != nil {
		return errors.New("invalid ZIP file")
	}
	defer reader.Close()
	for _, entry := range reader.File {
		clean := filepath.Clean(entry.Name)
		parts := strings.Split(filepath.ToSlash(clean), "/")
		if len(parts) > 0 && parts[0] == expectedRoot {
			parts = parts[1:]
		}
		if len(parts) == 0 {
			continue
		}
		target, err := safePath(destination, filepath.Join(parts...))
		if err != nil {
			return errors.New("ZIP contains an unsafe path")
		}
		if entry.FileInfo().IsDir() {
			if err = os.MkdirAll(target, 0750); err != nil {
				return err
			}
			continue
		}
		if entry.UncompressedSize64 > 512<<20 {
			return errors.New("ZIP entry is too large")
		}
		if err = os.MkdirAll(filepath.Dir(target), 0750); err != nil {
			return err
		}
		source, err := entry.Open()
		if err != nil {
			return err
		}
		output, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0640)
		if err != nil {
			source.Close()
			return err
		}
		_, copyErr := io.Copy(output, io.LimitReader(source, 513<<20))
		source.Close()
		output.Close()
		if copyErr != nil {
			return copyErr
		}
	}
	return nil
}

// inspectDataset 扫描目录结构，提取类别、训练划分和图片路径。
func inspectDataset(root string) ([]string, []imageRecord, error) {
	extensions := map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".bmp": true, ".tif": true, ".tiff": true, ".npy": true}
	labels := map[string]bool{}
	images := []imageRecord{}
	for _, split := range []string{"train", "test"} {
		splitRoot := filepath.Join(root, split)
		entries, err := os.ReadDir(splitRoot)
		if err != nil {
			return nil, nil, fmt.Errorf("dataset must contain %s directory", split)
		}
		for _, labelDir := range entries {
			if !labelDir.IsDir() {
				continue
			}
			labels[labelDir.Name()] = true
			count := 0
			_ = filepath.WalkDir(filepath.Join(splitRoot, labelDir.Name()), func(path string, entry os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if !entry.IsDir() && extensions[strings.ToLower(filepath.Ext(path))] && count < 10 {
					images = append(images, imageRecord{path: path, label: labelDir.Name(), split: split})
					count++
				}
				return nil
			})
		}
	}
	if len(labels) == 0 {
		return nil, nil, errors.New("no label folders found")
	}
	list := make([]string, 0, len(labels))
	for label := range labels {
		list = append(list, label)
	}
	sort.Strings(list)
	return list, images, nil
}
